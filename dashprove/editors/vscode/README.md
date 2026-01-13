# DashProve VS Code Extension

Language support for **USL** (Unified Specification Language) - formal verification made simple.

## Features

- **Syntax Highlighting**: Full TextMate grammar for USL files
- **IntelliSense**: Completions for keywords, types, and properties
- **Go to Definition**: Navigate to type and property definitions
- **Find References**: Find all usages of types and properties
- **Hover Information**: Type information and documentation on hover
- **Diagnostics**: Real-time error and warning reporting
- **Code Actions**: Quick fixes and refactoring suggestions
- **Code Lenses**: Verification actions inline with code
- **Formatting**: Automatic code formatting
- **Folding**: Collapse blocks, comments, and regions
- **Semantic Highlighting**: Context-aware token coloring
- **Inlay Hints**: Type annotations inline
- **Signature Help**: Parameter hints for contracts
- **Document Symbols**: Outline view of types and properties
- **Workspace Symbols**: Search across all USL files
- **Rename**: Rename types and properties across files
- **Selection Range**: Smart selection expansion
- **Call Hierarchy**: Navigate property references
- **Linked Editing**: Edit related identifiers simultaneously

## Requirements

- VS Code 1.85.0 or later
- DashProve LSP server (`dashprove-lsp`)

### Installing the LSP Server

```bash
# Build from source
cargo install --path crates/dashprove-lsp

# Or from crates.io (when published)
cargo install dashprove-lsp
```

## Extension Settings

This extension contributes the following settings:

- `dashprove.server.path`: Path to the dashprove-lsp binary
- `dashprove.server.args`: Additional arguments for the language server
- `dashprove.trace.server`: Trace communication with the server
- `dashprove.verification.autoVerify`: Automatically verify on save
- `dashprove.verification.timeout`: Verification timeout in milliseconds
- `dashprove.inlayHints.enabled`: Enable inlay hints
- `dashprove.codeLens.enabled`: Enable code lenses

## Commands

- `DashProve: Verify Current File` (Ctrl+Shift+V / Cmd+Shift+V)
- `DashProve: Verify Property at Cursor`
- `DashProve: Show Backend Information`
- `DashProve: Restart Language Server`

## USL Language Overview

USL (Unified Specification Language) is a specification language that compiles to multiple verification backends (Lean 4, TLA+, Kani, Alloy, Coq, Dafny, Isabelle).

### Example

```usl
// Type definitions
type User = { id: Int, name: String, email: String }
type Database = { users: Set<User> }

// Theorem (compiles to Lean 4, Coq)
theorem unique_ids {
    forall db: Database, u1: User, u2: User .
        u1 in db.users and u2 in db.users and u1.id == u2.id
        implies u1 == u2
}

// Contract (compiles to Kani)
contract add_user(db: Database, user: User) -> Result<Database> {
    requires {
        not exists u in db.users . u.id == user.id
    }
    ensures {
        user in result.users
        result.users.size() == db.users.size() + 1
    }
}

// Invariant (compiles to Alloy)
invariant no_duplicate_emails {
    forall db: Database, u1: User, u2: User .
        u1 in db.users and u2 in db.users and u1.email == u2.email
        implies u1 == u2
}

// Temporal property (compiles to TLA+)
temporal eventually_consistent {
    always(eventually(is_consistent(state)))
}
```

## Development

```bash
# Install dependencies
npm install

# Compile
npm run compile

# Watch mode
npm run watch

# Package
npm run package
```

## License

Apache-2.0
