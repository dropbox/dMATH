//! Moniker provider for USL symbols.
//!
//! Resolves stable identifiers for types and properties so editors can map
//! symbols across files or workspaces.

use crate::document::Document;
use tower_lsp::lsp_types::{Moniker, MonikerKind, Position, UniquenessLevel};

/// Resolve monikers for the symbol at the given position.
#[must_use]
pub fn resolve_monikers(doc: &Document, pos: Position) -> Option<Vec<Moniker>> {
    let word = doc.word_at_position(pos.line, pos.character)?;
    let spec = doc.spec.as_ref()?;

    let mut monikers = Vec::new();

    if spec.types.iter().any(|t| t.name == word) {
        monikers.push(make_moniker(word, "type"));
    }

    if spec.properties.iter().any(|p| p.name() == word) {
        monikers.push(make_moniker(word, "property"));
    }

    if monikers.is_empty() {
        None
    } else {
        Some(monikers)
    }
}

fn make_moniker(name: &str, kind: &str) -> Moniker {
    Moniker {
        scheme: "usl".to_string(),
        identifier: format!("{kind}:{name}"),
        unique: UniquenessLevel::Document,
        kind: Some(MonikerKind::Export),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::Url;

    #[test]
    fn resolves_type_moniker() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type User = { id: Int }".to_string(),
        );
        let pos = Position::new(0, 5);
        let monikers = resolve_monikers(&doc, pos).expect("expected moniker");

        assert_eq!(monikers.len(), 1);
        let moniker = &monikers[0];
        assert_eq!(moniker.scheme, "usl");
        assert_eq!(moniker.identifier, "type:User");
        assert_eq!(moniker.unique, UniquenessLevel::Document);
        assert_eq!(moniker.kind, Some(MonikerKind::Export));
    }

    #[test]
    fn resolves_property_moniker() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem user_valid { true }".to_string(),
        );
        let pos = Position::new(0, 8);
        let monikers = resolve_monikers(&doc, pos).expect("expected moniker");

        assert_eq!(monikers.len(), 1);
        let moniker = &monikers[0];
        assert_eq!(moniker.identifier, "property:user_valid");
    }

    #[test]
    fn returns_none_for_unknown_symbol() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem user_valid { true }".to_string(),
        );
        let pos = Position::new(0, 2); // on "th"
        assert!(resolve_monikers(&doc, pos).is_none());
    }
}
