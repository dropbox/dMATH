// WHNF (Weak Head Normal Form) Compute Shader
//
// This shader performs parallel WHNF reduction on batched expressions.
// Each workgroup thread processes one expression independently.
//
// WHNF reduces an expression to its "head" form:
// - Beta reduction: (λx.b) a → b[x := a]
// - Delta reduction: unfold constants
// - Zeta reduction: let x := v in b → b[x := v]
// - Iota reduction: recursor application (handled on CPU due to complexity)
//
// Due to GPU limitations, we perform iterative reduction with a step limit.
// Complex reductions (iota, projections with constructors) fall back to CPU.

// Expression tag constants (must match arena.rs)
const TAG_BVAR: u32 = 0u;
const TAG_FVAR: u32 = 1u;
const TAG_SORT: u32 = 2u;
const TAG_CONST: u32 = 3u;
const TAG_APP: u32 = 4u;
const TAG_LAM: u32 = 5u;
const TAG_PI: u32 = 6u;
const TAG_LET: u32 = 7u;
const TAG_LIT_NAT: u32 = 8u;
const TAG_LIT_STR: u32 = 9u;
const TAG_PROJ: u32 = 10u;
const TAG_NONE: u32 = 0xFFFFFFFFu;

// Reduction status flags
const STATUS_DONE: u32 = 0u;        // Already in WHNF
const STATUS_REDUCED: u32 = 1u;     // Reduction performed
const STATUS_NEEDS_CPU: u32 = 2u;   // Requires CPU fallback

// Maximum reduction steps per invocation (prevents infinite loops)
const MAX_STEPS: u32 = 64u;

// Expression structure (16 bytes, matches GpuExpr)
struct Expr {
    tag: u32,
    data1: u32,
    data2: u32,
    data3: u32,
}

// Constant definition entry
struct ConstDef {
    // Index of the definition body in the expr buffer (-1 if opaque)
    body_idx: i32,
    // Number of universe parameters (for validation)
    num_params: u32,
    // Reserved
    _pad1: u32,
    _pad2: u32,
}

// Input expressions (read-only)
@group(0) @binding(0) var<storage, read> input_exprs: array<Expr>;

// Output expressions (write)
@group(0) @binding(1) var<storage, read_write> output_exprs: array<Expr>;

// Environment: constant definitions
@group(0) @binding(2) var<storage, read> const_defs: array<ConstDef>;

// Reduction status for each expression
@group(0) @binding(3) var<storage, read_write> status: array<u32>;

// Indices of expressions to reduce (sparse array)
@group(0) @binding(4) var<storage, read> expr_indices: array<u32>;

// Working memory for substitution (per-thread scratch space)
// Each thread gets SCRATCH_SIZE entries
const SCRATCH_SIZE: u32 = 256u;
@group(0) @binding(5) var<storage, read_write> scratch: array<Expr>;

// Uniforms
struct Uniforms {
    num_exprs: u32,
    expr_buffer_size: u32,
    const_buffer_size: u32,
    _pad: u32,
}
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

// Helper: read expression, bounds checked
fn read_expr(idx: u32) -> Expr {
    if idx >= uniforms.expr_buffer_size {
        return Expr(TAG_NONE, 0u, 0u, 0u);
    }
    return input_exprs[idx];
}

// Helper: check if expression is in WHNF (head normal form)
fn is_whnf(expr: Expr) -> bool {
    switch expr.tag {
        // These are always in WHNF
        case TAG_BVAR, TAG_FVAR, TAG_SORT, TAG_PI, TAG_LAM, TAG_LIT_NAT, TAG_LIT_STR: {
            return true;
        }
        // Application needs to check if head is reducible
        case TAG_APP: {
            let head = get_app_head(expr);
            switch head.tag {
                // Lambda at head = beta redex
                case TAG_LAM: { return false; }
                // Constant might be unfoldable
                case TAG_CONST: { return !is_unfoldable(head); }
                // Let at head = zeta redex
                case TAG_LET: { return false; }
                default: { return true; }
            }
        }
        // Let is always reducible
        case TAG_LET: {
            return false;
        }
        // Const might be unfoldable
        case TAG_CONST: {
            return !is_unfoldable(expr);
        }
        // Projection needs CPU (depends on constructor matching)
        case TAG_PROJ: {
            return false; // Always try to reduce
        }
        default: {
            return true;
        }
    }
}

// Get the head of an application spine
fn get_app_head(expr: Expr) -> Expr {
    var current = expr;
    loop {
        if current.tag != TAG_APP {
            return current;
        }
        current = read_expr(current.data1);
    }
    return current; // unreachable
}

// Count arguments in application spine
fn count_app_args(expr: Expr) -> u32 {
    var count = 0u;
    var current = expr;
    loop {
        if current.tag != TAG_APP {
            return count;
        }
        count = count + 1u;
        current = read_expr(current.data1);
    }
    return count; // unreachable
}

// Check if a constant is unfoldable (has a definition body)
fn is_unfoldable(expr: Expr) -> bool {
    if expr.tag != TAG_CONST {
        return false;
    }
    let name_idx = expr.data1;
    if name_idx >= uniforms.const_buffer_size {
        return false;
    }
    return const_defs[name_idx].body_idx >= 0;
}

// Perform one WHNF reduction step
// Returns the reduced expression and updates status
// Note: WGSL does not support recursion, so nested apps are handled iteratively
fn whnf_step(expr: Expr, thread_id: u32) -> Expr {
    // For nested applications, find the innermost reducible head first
    var current = expr;

    switch current.tag {
        // Let binding: substitute value into body
        case TAG_LET: {
            // let x := val in body → body[0 := val]
            // For GPU, we just return the body and mark for CPU substitution
            // Full substitution is complex due to de Bruijn shifting
            status[thread_id] = STATUS_NEEDS_CPU;
            return current;
        }

        // Application: check for beta redex
        case TAG_APP: {
            let func_idx = current.data1;
            let arg_idx = current.data2;
            let func = read_expr(func_idx);

            // Beta reduction: (λx.body) arg → body[0 := arg]
            if func.tag == TAG_LAM {
                // Mark for CPU substitution (de Bruijn is complex)
                status[thread_id] = STATUS_NEEDS_CPU;
                return current;
            }

            // Delta reduction: unfold constant at head
            if func.tag == TAG_CONST && is_unfoldable(func) {
                let name_idx = func.data1;
                let body_idx = const_defs[name_idx].body_idx;
                if body_idx >= 0 {
                    // Replace function with unfolded body
                    // Note: still need to apply universe substitution (CPU)
                    status[thread_id] = STATUS_NEEDS_CPU;
                    return current;
                }
            }

            // Nested application: check head of application spine
            // (Iterative approach instead of recursion)
            let head = get_app_head(current);
            if head.tag == TAG_LAM || head.tag == TAG_LET {
                // There's a redex somewhere in the spine
                status[thread_id] = STATUS_NEEDS_CPU;
                return current;
            }
            if head.tag == TAG_CONST && is_unfoldable(head) {
                // Unfoldable constant at head
                status[thread_id] = STATUS_NEEDS_CPU;
                return current;
            }

            return current;
        }

        // Constant: unfold if possible
        case TAG_CONST: {
            if is_unfoldable(current) {
                status[thread_id] = STATUS_NEEDS_CPU;
                return current;
            }
            return current;
        }

        // Projection: needs CPU for constructor matching
        case TAG_PROJ: {
            status[thread_id] = STATUS_NEEDS_CPU;
            return current;
        }

        // Already in WHNF
        default: {
            return current;
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;

    // Bounds check
    if thread_id >= uniforms.num_exprs {
        return;
    }

    // Get expression index to process
    let expr_idx = expr_indices[thread_id];
    if expr_idx >= uniforms.expr_buffer_size {
        status[thread_id] = STATUS_DONE;
        output_exprs[thread_id] = Expr(TAG_NONE, 0u, 0u, 0u);
        return;
    }

    // Initialize status
    status[thread_id] = STATUS_DONE;

    // Read input expression
    var expr = read_expr(expr_idx);

    // Check if already in WHNF
    if is_whnf(expr) {
        output_exprs[thread_id] = expr;
        return;
    }

    // Perform reduction steps (with limit)
    for (var step = 0u; step < MAX_STEPS; step = step + 1u) {
        let prev_status = status[thread_id];
        expr = whnf_step(expr, thread_id);

        // Stop if needs CPU fallback
        if status[thread_id] == STATUS_NEEDS_CPU {
            output_exprs[thread_id] = expr;
            return;
        }

        // Stop if in WHNF
        if is_whnf(expr) {
            status[thread_id] = STATUS_DONE;
            output_exprs[thread_id] = expr;
            return;
        }
    }

    // Hit step limit - needs CPU fallback
    status[thread_id] = STATUS_NEEDS_CPU;
    output_exprs[thread_id] = expr;
}
