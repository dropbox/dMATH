# Lean5DB Search System Design

**Version:** 1.1
**Status:** Design (DO NOT EXECUTE)
**Author:** Claude
**Date:** 2026-01-07

## 1. Overview

This document specifies a search system for Lean5DB that provides:
1. **Keyword search** - Fast text matching on names, docstrings, types
2. **Semantic search** - Embedding-based similarity for finding related theorems
3. **Type search** - Pattern matching on types (like Lean 4's `#find`)
4. **Structural search** - Find expressions by AST patterns

Integration target: `git@github.com:dropbox/sg.git`

## 2. Lean 4's Existing Search

Lean 4/Mathlib provides limited search:

| Feature | Lean 4 `#find` | Lean5DB Target |
|---------|----------------|----------------|
| Type pattern | ✓ `#find _ + _ = _ + _` | ✓ Enhanced |
| Keyword | ✗ | ✓ Full-text |
| Semantic | ✗ | ✓ Embeddings |
| Structural | ✗ | ✓ AST patterns |
| Approximate | ✗ | ✓ Fuzzy matching |
| Ranking | ✗ | ✓ Relevance scoring |

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Search API (JSON-RPC)                       │
├─────────────────────────────────────────────────────────────────┤
│  Query Parser  │  Query Planner  │  Result Ranker  │  Cache     │
├─────────────────────────────────────────────────────────────────┤
│                      Search Indexes                              │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Keyword     │  Semantic    │  Type        │  Structural        │
│  (sg/tantivy)│  (vectors)   │  (trie)      │  (pattern db)      │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                      Lean5DB Storage                             │
│  Constants │ Types │ Expressions │ Docstrings │ Proofs          │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Search Types

### 4.1 Keyword Search

**Purpose:** Find constants by name, docstring, or type text.

**Index:**
- Constant names (tokenized: `Nat.add_comm` → `nat`, `add`, `comm`)
- Docstrings (full-text indexed)
- Type strings (rendered to text)
- Module paths

**Queries:**
```
keyword:comm add        # Find names containing "comm" and "add"
doc:"associative"       # Search docstrings
module:Mathlib.Algebra  # Filter by module
```

**Implementation:** Integrate with `sg` (sourcegraph) or use Tantivy (Rust).

### 4.2 Semantic Search

**Purpose:** Find theorems similar in meaning, even with different names.

**Approach:**
1. Generate embeddings for each constant:
   - Name + type rendered as natural language
   - Docstring (if present)
   - Proof structure summary
2. Store in vector index (HNSW)
3. Query with natural language or another theorem

**Embedding Sources:**
- Mathematical text: `∀ n m : Nat, n + m = m + n`
- Natural language: "addition of natural numbers is commutative"
- Mixed: Docstring + type signature

**Index Structure:**
```rust
struct SemanticIndex {
    /// HNSW index for fast approximate nearest neighbor
    hnsw: HnswIndex<f32>,

    /// Constant ID → embedding vector
    embeddings: Vec<[f32; EMBEDDING_DIM]>,  // EMBEDDING_DIM = 384 or 768

    /// Metadata for re-ranking
    constant_ids: Vec<ConstantId>,
}
```

**Queries:**
```
similar:Nat.add_comm           # Find theorems similar to this one
"commutative operations"       # Natural language query
semantic:"ring homomorphism"   # Explicit semantic search
```

### 4.3 Type Search (Enhanced `#find`)

**Purpose:** Find constants by type pattern, more powerful than Lean 4.

**Patterns:**
```
_ + _ = _ + _          # Commutativity pattern
?n + ?m = ?m + ?n      # Named wildcards (must match)
∀ _, _ → _             # Implications
List ?a → List ?a      # Polymorphic patterns
```

**Enhancements over Lean 4:**
- Fuzzy matching (allow extra quantifiers)
- Subexpression search (find pattern anywhere in type)
- Negation patterns (`NOT ∀ _`)
- Result type filtering (`returns:Bool`)

**Index Structure:**
```rust
struct TypeIndex {
    /// Type signature → constant IDs (exact match)
    exact: HashMap<TypeHash, Vec<ConstantId>>,

    /// Pattern trie for wildcard matching
    pattern_trie: PatternTrie,

    /// Structural fingerprints for approximate matching
    fingerprints: FingerprintIndex,
}
```

### 4.4 Structural Search

**Purpose:** Find expressions by AST structure.

**Patterns:**
```
app(_, const(Nat.add))           # Applications of Nat.add
lam(_, _, app(app(_, bvar(0)), _))  # λ x, f x _ pattern
forallE(_, sort(_), _)           # ∀ over Type
```

**Use Cases:**
- Find all uses of a specific constant
- Find all proofs using a specific tactic pattern
- Locate similar proof structures

## 5. Integration with `sg`

The `sg` (sourcegraph) repository provides existing search infrastructure.

### 5.1 Index Generation

Lean5DB generates index files compatible with `sg`:

```rust
/// Generate sg-compatible index from Lean5DB
pub fn export_to_sg(db: &Lean5Db, output: &Path) -> Result<()> {
    // 1. Export constants as "documents"
    let docs = db.constants().map(|c| SgDocument {
        id: c.name.to_string(),
        content: format_constant_for_search(c),
        metadata: SgMetadata {
            kind: c.kind.to_string(),
            module: c.module_path(),
            type_sig: c.type_string(),
        },
    });

    // 2. Write to sg index format
    sg::IndexWriter::new(output)?.write_documents(docs)?;

    Ok(())
}
```

### 5.2 Query Translation

```rust
/// Translate Lean5DB query to sg query
pub fn to_sg_query(q: &SearchQuery) -> SgQuery {
    match q {
        SearchQuery::Keyword { terms, filters } => {
            SgQuery::text(terms)
                .filter("kind", filters.kind)
                .filter("module", filters.module_prefix)
        }
        SearchQuery::Semantic { .. } => {
            // sg doesn't support semantic - use local index
            panic!("Semantic search not supported by sg")
        }
        // ...
    }
}
```

### 5.3 Hybrid Search

For best results, combine `sg` keyword search with local semantic search:

```rust
pub async fn hybrid_search(
    query: &str,
    sg_client: &SgClient,
    semantic_index: &SemanticIndex,
    k: usize,
) -> Vec<SearchResult> {
    // Run both searches in parallel
    let (keyword_results, semantic_results) = tokio::join!(
        sg_client.search(query, k * 2),
        semantic_index.search(query, k * 2),
    );

    // Reciprocal rank fusion
    let mut scores: HashMap<ConstantId, f32> = HashMap::new();

    for (rank, result) in keyword_results.iter().enumerate() {
        *scores.entry(result.id).or_default() += 1.0 / (60.0 + rank as f32);
    }
    for (rank, result) in semantic_results.iter().enumerate() {
        *scores.entry(result.id).or_default() += 1.0 / (60.0 + rank as f32);
    }

    // Sort by combined score
    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    results.into_iter().take(k).map(|(id, score)| {
        SearchResult { id, score, /* ... */ }
    }).collect()
}
```

## 6. Index Building

### 6.1 Build Pipeline

```
.olean files → Lean5DB → Search Indexes
                  ↓
            ┌─────┴─────┐
            ↓           ↓
      Keyword Index  Semantic Index
      (sg format)    (HNSW vectors)
            ↓           ↓
      Type Index    Structural Index
```

### 6.2 Embedding Generation

Options for generating embeddings:

| Model | Dim | Speed | Quality | Notes |
|-------|-----|-------|---------|-------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose |
| text-embedding-3-small | 1536 | API | Better | OpenAI API |
| Mathbert | 768 | Med | Best for math | Specialized |
| E5-large-v2 | 1024 | Slow | Excellent | Best general |

**Recommended:** Start with `all-MiniLM-L6-v2` locally, option to use API for better quality.

### 6.3 Index Sizes (Mathlib estimate)

| Index | Size | Load Time |
|-------|------|-----------|
| Keyword (sg) | ~50 MB | ~100ms |
| Semantic (384d) | ~400 MB | ~200ms |
| Type patterns | ~80 MB | ~50ms |
| Structural | ~120 MB | ~80ms |
| **Total** | **~650 MB** | **~430ms** |

## 7. Query Language

### 7.1 Grammar

```ebnf
query      = term+
term       = keyword_term | type_term | semantic_term | filter
keyword_term = WORD | QUOTED_STRING
type_term  = "type:" pattern
semantic_term = "similar:" const_name | "semantic:" QUOTED_STRING
filter     = filter_key ":" filter_value
filter_key = "module" | "kind" | "has_proof" | "returns"
pattern    = lean_type_pattern
```

### 7.2 Examples

```
# Keyword search
comm add nat

# Type pattern search
type:∀ n m : Nat, n + _ = _ + n

# Semantic search
similar:Nat.add_comm
semantic:"list concatenation is associative"

# Combined with filters
comm module:Mathlib.Algebra kind:theorem

# Complex query
(type:_ * _ = _ * _) OR semantic:"multiplication commutes" module:Mathlib
```

## 8. API Design

### 8.1 JSON-RPC Methods

```typescript
// Search request
interface SearchRequest {
  query: string;
  max_results?: number;  // default: 20
  search_types?: ("keyword" | "semantic" | "type" | "structural")[];
  filters?: {
    modules?: string[];
    kinds?: ("theorem" | "def" | "axiom" | "structure")[];
    has_proof?: boolean;
  };
}

// Search response
interface SearchResponse {
  results: SearchResult[];
  total_count: number;
  search_time_ms: number;
}

interface SearchResult {
  constant_name: string;
  module: string;
  kind: string;
  type_signature: string;
  docstring?: string;
  score: number;
  match_type: "keyword" | "semantic" | "type" | "structural";
  highlights?: Highlight[];  // For keyword matches
}
```

### 8.2 Rust API

```rust
pub struct SearchEngine {
    keyword_index: KeywordIndex,
    semantic_index: SemanticIndex,
    type_index: TypeIndex,
    structural_index: StructuralIndex,
    db: Arc<Lean5Db>,
}

impl SearchEngine {
    pub fn search(&self, query: &SearchQuery) -> SearchResults;
    pub fn similar(&self, constant: &Name, k: usize) -> Vec<ConstantId>;
    pub fn find_by_type(&self, pattern: &TypePattern) -> Vec<ConstantId>;
    pub fn find_usages(&self, constant: &Name) -> Vec<Usage>;
}
```

## 9. Incremental Updates

When Mathlib updates, we need incremental index updates:

```rust
pub struct IndexUpdater {
    /// Content hashes from last index build
    previous_hashes: HashMap<ConstantId, [u8; 32]>,
}

impl IndexUpdater {
    pub fn update(&mut self, db: &Lean5Db, indexes: &mut SearchIndexes) -> Stats {
        let mut added = 0;
        let mut modified = 0;
        let mut deleted = 0;

        for constant in db.constants() {
            let hash = constant.content_hash();
            match self.previous_hashes.get(&constant.id) {
                None => {
                    // New constant
                    indexes.add(constant);
                    added += 1;
                }
                Some(prev_hash) if *prev_hash != hash => {
                    // Modified constant
                    indexes.update(constant);
                    modified += 1;
                }
                Some(_) => {
                    // Unchanged
                }
            }
        }

        // Find deleted constants
        for (id, _) in &self.previous_hashes {
            if !db.has_constant(*id) {
                indexes.remove(*id);
                deleted += 1;
            }
        }

        Stats { added, modified, deleted }
    }
}
```

## 10. Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Keyword search | <10ms | Using sg/tantivy |
| Semantic search | <50ms | HNSW approximate |
| Type pattern search | <20ms | Pattern trie |
| Structural search | <30ms | Fingerprint filter + verify |
| Combined search | <100ms | Parallel execution |
| Index load | <500ms | Memory mapped |
| Incremental update | <5s | For typical Mathlib PR |

## 11. Implementation Phases

### Phase 1: Keyword Search (~2 weeks)
- [ ] Export Lean5DB constants to sg-compatible format
- [ ] Integrate sg search API
- [ ] Basic query parsing
- [ ] Result formatting

### Phase 2: Type Search (~2 weeks)
- [ ] Type pattern parser
- [ ] Pattern trie index
- [ ] Wildcard matching
- [ ] Enhanced patterns (subexpr, negation)

### Phase 3: Semantic Search (~3 weeks)
- [ ] Embedding generation pipeline
- [ ] HNSW index implementation or integration
- [ ] Natural language query support
- [ ] Similar theorem lookup

### Phase 4: Structural Search (~2 weeks)
- [ ] Expression pattern language
- [ ] Structural fingerprinting
- [ ] Pattern matching engine
- [ ] Usage finder

### Phase 5: Integration (~1 week)
- [ ] Unified query language
- [ ] Hybrid search ranking
- [ ] JSON-RPC API
- [ ] CLI interface

**Total: ~10 weeks**

## 12. Flaws and Risks

### 12.1 Critical Flaws (Must Address)

1. **Embedding quality for math** - General-purpose embeddings (MiniLM, E5) are trained on natural language, not mathematical notation. They may cluster `Nat.add_comm` and `Int.add_comm` far apart while placing unrelated concepts with similar words together.
   - **Mitigation:** Fine-tune embeddings on Mathlib-specific corpus, or use math-specialized models like Mathbert. Include ablation study in Phase 3.

2. **Index-source version coupling** - No mechanism to verify search index matches Lean5DB version. Stale indexes return incorrect results silently.
   - **Mitigation:** Embed Lean5DB content hash in index header. Reject queries if hash mismatch.

3. **Type unification complexity** - Type pattern search with named wildcards (`?n + ?m = ?m + ?n`) requires unification, which is undecidable in general. Pathological patterns could hang.
   - **Mitigation:** Timeout on unification (100ms), limit pattern depth, restrict to decidable fragment.

4. **Memory pressure from semantic index** - 130K constants × 384 floats × 4 bytes = ~200MB just for vectors, plus HNSW graph structure (~2-3× vectors). Full Mathlib could require 500MB+ RAM.
   - **Mitigation:** Memory-mapped index, tiered loading (hot theorems in RAM, cold on disk), quantization (int8 reduces to ~60MB).

5. **Query injection** - User-provided patterns are parsed and executed. Malformed patterns could exploit parser bugs or cause resource exhaustion.
   - **Mitigation:** Strict grammar, resource limits, sandboxed evaluation.

### 12.2 Significant Flaws (Should Address)

6. **Tokenization ambiguity** - `Nat.add_comm` could tokenize as `[nat, add, comm]` or `[nat.add, comm]` or `[nat.add.comm]`. Different tokenizations give different search results.
   - **Mitigation:** Define canonical tokenization in spec. Support multiple tokenization strategies with explicit selection.

7. **No search history / personalization** - All users get same results. No learning from past searches or user preferences.
   - **Mitigation:** Optional user profile with search history, clicked results. Use for re-ranking in future phase.

8. **Embedding model dependency** - If embedding model changes or becomes unavailable, all semantic indexes become invalid. Re-embedding 130K constants is expensive.
   - **Mitigation:** Store model identifier + version in index. Support multiple embedding models simultaneously.

9. **No negative search** - Can't express "theorems about groups but NOT abelian groups" or "definitions without proofs."
   - **Mitigation:** Add NOT/MINUS operators to query language. Filter post-retrieval for semantic negation.

10. **Cross-reference blindness** - Search doesn't know that `Nat.add_comm` is used by `Nat.add_assoc`. Can't answer "what uses this theorem?"
    - **Mitigation:** Build dependency graph during indexing. Add `uses:` and `usedby:` filters.

### 12.3 Minor Flaws (Can Defer)

11. **Unicode handling** - Mathematical symbols (∀, →, ℕ) may not search correctly. User types "forall" but index has "∀".
    - **Mitigation:** Normalize Unicode to ASCII aliases during indexing. Support both forms in queries.

12. **No faceted search** - Can't browse by category (all ring theorems, all topology lemmas).
    - **Mitigation:** Extract Mathlib hierarchy as facets. Add browse-by-category API.

13. **Approximate match opacity** - When fuzzy matching returns results, user doesn't know why. "Did it match the name or type?"
    - **Mitigation:** Include match explanation in results. Highlight matched terms.

14. **No search federation** - Can only search one Lean5DB instance. Can't search Mathlib + Std + custom project together.
    - **Mitigation:** Support multiple index sources with federated query routing.

15. **Stale embedding cache** - If constant changes but embedding isn't regenerated, semantic search returns wrong results.
    - **Mitigation:** Content hash → embedding hash mapping. Invalidate on mismatch.

## 13. Improvements

### 13.1 High-Value Improvements (Include in v1)

1. **Proof strategy indexing** - Index the proof term structure to enable "find proofs by induction" or "find proofs using specific lemmas."
   ```rust
   struct ProofIndex {
       tactics_used: HashMap<TacticId, Vec<ConstantId>>,
       lemmas_applied: HashMap<ConstantId, Vec<ConstantId>>,
       proof_depth: HashMap<ConstantId, u32>,
   }
   ```

2. **Trigram index for fuzzy name search** - Enable typo-tolerant search using trigram similarity.
   - `add_com` matches `add_comm` (edit distance 1)
   - `comutative` matches `commutative`
   - Fast with precomputed trigram index

3. **Type-aware semantic embeddings** - Embed types structurally, not just as text. `∀ a : Type, a → a` and `∀ b : Type, b → b` should have identical embeddings.
   ```rust
   fn structural_type_embedding(ty: &Expr) -> Vec<f32> {
       // Normalize variable names, then embed
       let normalized = alpha_normalize(ty);
       embed_expression(normalized)
   }
   ```

4. **Cached query plans** - Parse and optimize queries once, reuse for repeated searches.
   ```rust
   struct QueryCache {
       parsed: LruCache<String, ParsedQuery>,
       plans: LruCache<QueryHash, QueryPlan>,
   }
   ```

5. **Streaming results** - Return results as they're found, don't wait for all indexes.
   ```rust
   pub fn search_streaming(&self, query: &SearchQuery) -> impl Stream<Item = SearchResult>;
   ```

### 13.2 Medium-Value Improvements (Include in v1.1)

6. **Example-based search** - "Find theorems like this one" with automatic feature extraction.
   ```rust
   pub fn search_by_example(&self, example: &ConstantInfo, k: usize) -> Vec<SearchResult> {
       let features = extract_features(example);
       self.multi_index_search(features, k)
   }
   ```

7. **Search result clustering** - Group similar results to avoid redundancy.
   - 10 variants of `add_comm` for different types → show one, expandable to see variants

8. **Natural language type description** - Convert type `∀ n m : Nat, n + m = m + n` to "for all natural numbers n and m, n plus m equals m plus n" for better semantic search.

9. **Search analytics** - Track what users search for, what they click. Use to improve ranking.
   ```rust
   struct SearchAnalytics {
       queries: Vec<QueryLog>,
       clicks: Vec<ClickLog>,
       // Aggregate for ML ranking
   }
   ```

10. **Autocompletion** - As user types, suggest completions from index.
    - `Nat.add_` → `[Nat.add_comm, Nat.add_assoc, Nat.add_zero, ...]`

### 13.3 Future Improvements (Post v1.1)

11. **Proof search by goal** - Given a goal type, find lemmas that could make progress.
    ```rust
    pub fn search_for_goal(&self, goal: &Expr) -> Vec<ApplicableLemma>;
    ```

12. **Analogy search** - "Ring : RingHom :: Group : ?" → `GroupHom`
    ```rust
    pub fn search_by_analogy(&self, a: &Name, b: &Name, c: &Name) -> Vec<ConstantId>;
    ```

13. **Multi-modal search** - Search using LaTeX rendered images, handwritten math OCR.

14. **Federated search** - Search across multiple Lean5DB instances (Mathlib, Std, custom libs) with merged ranking.

15. **Learned ranking** - Train ranking model on user click data to improve result ordering.

## 14. Addressed Flaws Summary

| Flaw | Severity | Mitigation | Phase |
|------|----------|------------|-------|
| Math embedding quality | Critical | Fine-tune or use Mathbert | 3 |
| Index version coupling | Critical | Content hash in header | 1 |
| Type unification complexity | Critical | Timeout, depth limit | 2 |
| Memory pressure | Critical | Mmap, quantization | 3 |
| Query injection | Critical | Strict grammar, limits | 1 |
| Tokenization ambiguity | Significant | Canonical spec | 1 |
| No personalization | Significant | User profiles | Future |
| Model dependency | Significant | Multi-model support | 3 |
| No negative search | Significant | NOT operator | 2 |
| Cross-reference blindness | Significant | Dependency graph | 2 |
| Unicode handling | Minor | Normalization | 1 |
| No faceted search | Minor | Category facets | Future |
| Match opacity | Minor | Explanations | 2 |
| No federation | Minor | Multi-source | Future |
| Stale embeddings | Minor | Hash validation | 3 |

## 15. Future Enhancements

1. **Proof search** - Find proofs that use specific tactics or lemmas
2. **Counterexample search** - Find constants that violate a given property
3. **Analogy search** - "Find theorems about X that are like Y for Z"
4. **Learning to rank** - Use ML to improve result ranking based on usage
5. **Cross-library search** - Search across multiple Lean projects
6. **IDE integration** - Real-time search as you type in VS Code / Emacs
7. **Natural language interface** - "Find all theorems about prime numbers"
8. **Citation graph** - Which papers cite which theorems, trace intellectual lineage

---

## Appendix A: sg Integration Details

The `sg` repository structure (assumed):

```
sg/
├── src/
│   ├── index/        # Indexing infrastructure
│   ├── query/        # Query parsing and execution
│   └── api/          # Search API
├── indexes/          # Index storage
└── config/           # Configuration
```

Lean5DB will provide:
- Index exporter compatible with sg format
- Custom tokenizer for Lean syntax
- Query translator for Lean-specific patterns

---

## Appendix B: Embedding Text Templates

For generating embeddings, constants are rendered to natural language:

**Theorem:**
```
Theorem Nat.add_comm states that for all natural numbers n and m,
n plus m equals m plus n. This is the commutativity property of
natural number addition.
```

**Definition:**
```
Definition List.map takes a function f from A to B and a list of A,
and returns a list of B by applying f to each element.
```

**Structure:**
```
Structure Ring has fields: carrier (a Type), add and mul (binary operations),
zero and one (identity elements), with axioms for associativity, commutativity,
and distributivity.
```
