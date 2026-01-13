//! Proof pattern extraction and matching
//!
//! This module identifies common tactic sequences (patterns) from successful proofs
//! and uses them to suggest repair strategies for failing proofs.
//!
//! # Architecture
//!
//! The pattern system works in three phases:
//!
//! 1. **Extraction**: Analyze successful proof tactic sequences to identify
//!    frequently-occurring subsequences (n-grams of tactics)
//!
//! 2. **Indexing**: Store patterns with their contexts, success counts, and
//!    typical applications (property types, complexity levels)
//!
//! 3. **Matching**: Given a failing proof state, find relevant patterns that
//!    might complete the proof based on current context and partial progress
//!
//! # Example
//!
//! ```rust
//! use dashprove_learning::patterns::{PatternExtractor, PatternDatabase, TacticSequence};
//!
//! let mut extractor = PatternExtractor::new(3); // max pattern length 3
//!
//! // Record successful tactic sequences
//! extractor.record_sequence(&["intro", "simp", "decide"]);
//! extractor.record_sequence(&["intro", "simp", "trivial"]);
//! extractor.record_sequence(&["intro", "simp", "rfl"]);
//!
//! // Extract common patterns
//! let patterns = extractor.extract_patterns(2); // min frequency 2
//!
//! // "intro, simp" appears in all three sequences
//! assert!(patterns.iter().any(|p| p.sequence.tactics == vec!["intro", "simp"]));
//! ```

use crate::tactics::TacticContext;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A sequence of tactics (n-gram)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TacticSequence {
    /// Ordered list of tactics
    pub tactics: Vec<String>,
}

impl TacticSequence {
    /// Create a new sequence from tactics
    pub fn new(tactics: Vec<String>) -> Self {
        Self { tactics }
    }

    /// Create from a slice of string references
    pub fn from_slice(tactics: &[&str]) -> Self {
        Self {
            tactics: tactics.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Get the length of the sequence
    pub fn len(&self) -> usize {
        self.tactics.len()
    }

    /// Check if sequence is empty
    pub fn is_empty(&self) -> bool {
        self.tactics.is_empty()
    }

    /// Check if this sequence is a prefix of another
    pub fn is_prefix_of(&self, other: &TacticSequence) -> bool {
        if self.tactics.len() > other.tactics.len() {
            return false;
        }
        self.tactics
            .iter()
            .zip(other.tactics.iter())
            .all(|(a, b)| a == b)
    }

    /// Check if this sequence contains another as a subsequence
    pub fn contains(&self, other: &TacticSequence) -> bool {
        if other.tactics.is_empty() {
            return true;
        }
        if other.tactics.len() > self.tactics.len() {
            return false;
        }
        self.tactics
            .windows(other.tactics.len())
            .any(|window| window == other.tactics.as_slice())
    }
}

impl std::fmt::Display for TacticSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tactics.join(" → "))
    }
}

/// Statistics about a pattern's usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternStats {
    /// Number of times this pattern led to successful proof
    pub success_count: u32,
    /// Number of times this pattern was followed by failure
    pub failure_count: u32,
    /// Contexts where this pattern was successful
    pub successful_contexts: Vec<TacticContext>,
    /// Average position in the tactic sequence where this pattern appears
    pub avg_position: f64,
}

impl PatternStats {
    /// Success rate for this pattern
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f64 / total as f64
        }
    }

    /// Total observations
    pub fn total(&self) -> u32 {
        self.success_count + self.failure_count
    }

    /// Confidence-adjusted score (Wilson score lower bound)
    pub fn wilson_score(&self) -> f64 {
        let n = self.total() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let p = self.success_rate();
        let z = 1.96; // 95% confidence

        let denominator = 1.0 + z * z / n;
        let center = p + z * z / (2.0 * n);
        let adjustment = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt();

        (center - adjustment) / denominator
    }
}

/// A proof pattern with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofPattern {
    /// The tactic sequence
    pub sequence: TacticSequence,
    /// Usage statistics
    pub stats: PatternStats,
    /// Common follow-up tactics after this pattern
    pub continuations: HashMap<String, u32>,
    /// Pattern description (auto-generated or user-provided)
    pub description: Option<String>,
}

impl ProofPattern {
    /// Create a new pattern
    pub fn new(sequence: TacticSequence) -> Self {
        Self {
            sequence,
            stats: PatternStats::default(),
            continuations: HashMap::new(),
            description: None,
        }
    }

    /// Get the most common continuation tactic
    pub fn best_continuation(&self) -> Option<(&String, u32)> {
        self.continuations
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(tactic, count)| (tactic, *count))
    }

    /// Get top N continuations
    pub fn top_continuations(&self, n: usize) -> Vec<(&String, u32)> {
        let mut sorted: Vec<_> = self.continuations.iter().map(|(t, c)| (t, *c)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(n).collect()
    }
}

/// Extracts patterns from tactic sequences
pub struct PatternExtractor {
    /// Maximum pattern length to extract
    max_length: usize,
    /// Recorded sequences for extraction
    sequences: Vec<(TacticSequence, Option<TacticContext>)>,
    /// N-gram counts
    ngram_counts: HashMap<TacticSequence, u32>,
}

impl PatternExtractor {
    /// Create a new extractor with maximum pattern length
    pub fn new(max_length: usize) -> Self {
        Self {
            max_length: max_length.max(1),
            sequences: Vec::new(),
            ngram_counts: HashMap::new(),
        }
    }

    /// Record a successful tactic sequence
    pub fn record_sequence(&mut self, tactics: &[&str]) {
        self.record_sequence_with_context(tactics, None);
    }

    /// Record a tactic sequence with context
    pub fn record_sequence_with_context(
        &mut self,
        tactics: &[&str],
        context: Option<TacticContext>,
    ) {
        let seq = TacticSequence::from_slice(tactics);
        self.sequences.push((seq.clone(), context));

        // Extract all n-grams up to max_length
        for n in 1..=self.max_length.min(tactics.len()) {
            for window in tactics.windows(n) {
                let ngram = TacticSequence::from_slice(window);
                *self.ngram_counts.entry(ngram).or_insert(0) += 1;
            }
        }
    }

    /// Extract patterns that occur at least min_frequency times
    pub fn extract_patterns(&self, min_frequency: u32) -> Vec<ProofPattern> {
        let mut patterns: Vec<ProofPattern> = self
            .ngram_counts
            .iter()
            .filter(|(_, count)| **count >= min_frequency)
            .map(|(seq, count)| {
                let mut pattern = ProofPattern::new(seq.clone());
                pattern.stats.success_count = *count;

                // Find continuations for this pattern
                self.compute_continuations(&mut pattern);

                // Compute average position
                pattern.stats.avg_position = self.compute_avg_position(seq);

                pattern
            })
            .collect();

        // Sort by frequency (descending) then by length (descending)
        patterns.sort_by(|a, b| {
            let count_cmp = b.stats.success_count.cmp(&a.stats.success_count);
            if count_cmp == std::cmp::Ordering::Equal {
                b.sequence.len().cmp(&a.sequence.len())
            } else {
                count_cmp
            }
        });

        patterns
    }

    /// Compute continuation tactics for a pattern
    fn compute_continuations(&self, pattern: &mut ProofPattern) {
        for (seq, _ctx) in &self.sequences {
            // Find where this pattern appears in the sequence
            let pattern_len = pattern.sequence.tactics.len();
            if pattern_len >= seq.tactics.len() {
                continue;
            }

            for (i, window) in seq.tactics.windows(pattern_len).enumerate() {
                if window == pattern.sequence.tactics.as_slice() {
                    // Check if there's a next tactic
                    if i + pattern_len < seq.tactics.len() {
                        let next = &seq.tactics[i + pattern_len];
                        *pattern.continuations.entry(next.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    /// Compute average position where pattern appears
    fn compute_avg_position(&self, pattern: &TacticSequence) -> f64 {
        let mut total_position = 0.0;
        let mut count = 0.0;

        for (seq, _ctx) in &self.sequences {
            if seq.tactics.len() < pattern.tactics.len() {
                continue;
            }

            for (i, window) in seq.tactics.windows(pattern.tactics.len()).enumerate() {
                if window == pattern.tactics.as_slice() {
                    // Normalize position to [0, 1] range
                    let normalized = if seq.tactics.len() <= 1 {
                        0.0
                    } else {
                        i as f64 / (seq.tactics.len() - 1) as f64
                    };
                    total_position += normalized;
                    count += 1.0;
                }
            }
        }

        if count == 0.0 {
            0.0
        } else {
            total_position / count
        }
    }

    /// Get the number of recorded sequences
    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    /// Get the number of unique n-grams
    pub fn ngram_count(&self) -> usize {
        self.ngram_counts.len()
    }

    /// Clear all recorded data
    pub fn clear(&mut self) {
        self.sequences.clear();
        self.ngram_counts.clear();
    }
}

impl Default for PatternExtractor {
    fn default() -> Self {
        Self::new(5) // Default max length of 5
    }
}

/// Database of proof patterns for lookup and suggestion
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PatternDatabase {
    /// Patterns indexed by first tactic
    by_first_tactic: HashMap<String, Vec<ProofPattern>>,
    /// Patterns indexed by context type
    by_context: HashMap<String, Vec<ProofPattern>>,
    /// All patterns
    all_patterns: Vec<ProofPattern>,
}

impl PatternDatabase {
    /// Create a new empty database
    pub fn new() -> Self {
        Self::default()
    }

    /// Build database from extracted patterns
    pub fn from_patterns(patterns: Vec<ProofPattern>) -> Self {
        let mut db = Self::new();

        for pattern in patterns {
            // Index by first tactic
            if let Some(first) = pattern.sequence.tactics.first() {
                db.by_first_tactic
                    .entry(first.clone())
                    .or_default()
                    .push(pattern.clone());
            }

            // Index by contexts where pattern was successful
            for ctx in &pattern.stats.successful_contexts {
                db.by_context
                    .entry(ctx.property_type.clone())
                    .or_default()
                    .push(pattern.clone());
            }

            db.all_patterns.push(pattern);
        }

        db
    }

    /// Add a single pattern to the database
    pub fn add_pattern(&mut self, pattern: ProofPattern) {
        // Index by first tactic
        if let Some(first) = pattern.sequence.tactics.first() {
            self.by_first_tactic
                .entry(first.clone())
                .or_default()
                .push(pattern.clone());
        }

        // Index by contexts
        for ctx in &pattern.stats.successful_contexts {
            self.by_context
                .entry(ctx.property_type.clone())
                .or_default()
                .push(pattern.clone());
        }

        self.all_patterns.push(pattern);
    }

    /// Find patterns that start with the given tactic
    pub fn patterns_starting_with(&self, tactic: &str) -> &[ProofPattern] {
        self.by_first_tactic
            .get(tactic)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Find patterns for a given context
    pub fn patterns_for_context(&self, context: &TacticContext) -> Vec<&ProofPattern> {
        self.by_context
            .get(&context.property_type)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Find patterns that could complete a partial sequence
    ///
    /// Given current tactics, find patterns where the current tactics
    /// are a prefix, and return suggested continuations.
    pub fn suggest_continuations(&self, current: &[&str], n: usize) -> Vec<SuggestedContinuation> {
        let current_seq = TacticSequence::from_slice(current);
        let mut suggestions: Vec<SuggestedContinuation> = Vec::new();

        for pattern in &self.all_patterns {
            if current_seq.is_prefix_of(&pattern.sequence) && current.len() < pattern.sequence.len()
            {
                // This pattern extends the current sequence
                let next_tactics: Vec<String> = pattern.sequence.tactics[current.len()..].to_vec();
                let confidence = pattern.stats.wilson_score();

                suggestions.push(SuggestedContinuation {
                    tactics: next_tactics,
                    pattern_name: pattern.description.clone(),
                    confidence,
                    pattern_frequency: pattern.stats.success_count,
                });
            }
        }

        // Sort by confidence
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        suggestions.into_iter().take(n).collect()
    }

    /// Get total pattern count
    pub fn len(&self) -> usize {
        self.all_patterns.len()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.all_patterns.is_empty()
    }

    /// Get all patterns
    pub fn patterns(&self) -> &[ProofPattern] {
        &self.all_patterns
    }

    /// Save database to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::LearningError> {
        crate::io::write_json_atomic(path, self)
    }

    /// Load database from file
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        crate::io::read_json(path)
    }

    /// Load database if exists, otherwise return empty
    pub fn load_or_default<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::LearningError> {
        match std::fs::File::open(path.as_ref()) {
            Ok(_) => Self::load_from_file(path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }
}

/// A suggested continuation for a partial proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedContinuation {
    /// Tactics to try next
    pub tactics: Vec<String>,
    /// Name of the pattern (if available)
    pub pattern_name: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// How often this pattern has succeeded
    pub pattern_frequency: u32,
}

impl std::fmt::Display for SuggestedContinuation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tactics.join(" → "))?;
        if let Some(ref name) = self.pattern_name {
            write!(f, " ({})", name)?;
        }
        write!(f, " [{:.1}% confidence]", self.confidence * 100.0)
    }
}

/// Proof repair system using patterns
pub struct ProofRepairer {
    /// Pattern database
    patterns: PatternDatabase,
    /// Maximum suggestions to generate
    max_suggestions: usize,
}

impl ProofRepairer {
    /// Create a new proof repairer with a pattern database
    pub fn new(patterns: PatternDatabase) -> Self {
        Self {
            patterns,
            max_suggestions: 5,
        }
    }

    /// Set maximum number of suggestions
    pub fn with_max_suggestions(mut self, max: usize) -> Self {
        self.max_suggestions = max;
        self
    }

    /// Suggest repairs for a failing proof
    ///
    /// Given the tactics tried so far, suggest alternative or continuation tactics
    /// based on patterns that worked in similar situations.
    pub fn suggest_repairs(
        &self,
        tried_tactics: &[&str],
        context: &TacticContext,
    ) -> Vec<RepairSuggestion> {
        let mut suggestions = Vec::new();

        // Strategy 1: Continue from current point
        let continuations = self
            .patterns
            .suggest_continuations(tried_tactics, self.max_suggestions);
        for cont in continuations {
            suggestions.push(RepairSuggestion {
                strategy: RepairStrategy::Continue,
                tactics: cont.tactics,
                confidence: cont.confidence,
                explanation: format!(
                    "Continue with pattern that succeeded {} times",
                    cont.pattern_frequency
                ),
            });
        }

        // Strategy 2: Backtrack and try alternative
        if !tried_tactics.is_empty() {
            let prefix = &tried_tactics[..tried_tactics.len() - 1];
            let alternatives = self
                .patterns
                .suggest_continuations(prefix, self.max_suggestions);
            for alt in alternatives {
                // Skip if it's the same as what we already tried
                if !alt.tactics.is_empty()
                    && alt.tactics[0] == tried_tactics[tried_tactics.len() - 1]
                {
                    continue;
                }
                suggestions.push(RepairSuggestion {
                    strategy: RepairStrategy::Backtrack(1),
                    tactics: alt.tactics,
                    confidence: alt.confidence * 0.9, // Slightly lower confidence for backtracking
                    explanation: "Backtrack one step and try alternative".to_string(),
                });
            }
        }

        // Strategy 3: Start fresh with patterns for this context
        let context_patterns = self.patterns.patterns_for_context(context);
        for pattern in context_patterns.into_iter().take(self.max_suggestions) {
            // Skip patterns that start the same way we already tried
            if !tried_tactics.is_empty()
                && !pattern.sequence.tactics.is_empty()
                && pattern.sequence.tactics[0] == tried_tactics[0]
            {
                continue;
            }
            suggestions.push(RepairSuggestion {
                strategy: RepairStrategy::Restart,
                tactics: pattern.sequence.tactics.clone(),
                confidence: pattern.stats.wilson_score() * 0.7, // Lower confidence for restart
                explanation: format!(
                    "Restart with pattern for {} properties",
                    context.property_type
                ),
            });
        }

        // Sort by confidence and deduplicate
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.dedup_by(|a, b| a.tactics == b.tactics);

        suggestions.into_iter().take(self.max_suggestions).collect()
    }

    /// Get the pattern database
    pub fn patterns(&self) -> &PatternDatabase {
        &self.patterns
    }
}

/// Strategy for repairing a proof
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairStrategy {
    /// Continue from current point with new tactics
    Continue,
    /// Backtrack N steps and try alternative
    Backtrack(usize),
    /// Start over with a completely different approach
    Restart,
}

impl std::fmt::Display for RepairStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepairStrategy::Continue => write!(f, "continue"),
            RepairStrategy::Backtrack(n) => write!(f, "backtrack {}", n),
            RepairStrategy::Restart => write!(f, "restart"),
        }
    }
}

/// A suggested repair for a failing proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairSuggestion {
    /// The repair strategy
    pub strategy: RepairStrategy,
    /// Tactics to try
    pub tactics: Vec<String>,
    /// Confidence in this suggestion (0.0 to 1.0)
    pub confidence: f64,
    /// Human-readable explanation
    pub explanation: String,
}

impl std::fmt::Display for RepairSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} ({:.1}%): {}",
            self.strategy,
            self.tactics.join(" → "),
            self.confidence * 100.0,
            self.explanation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tactics::ContextComplexity;

    #[test]
    fn test_tactic_sequence_creation() {
        let seq = TacticSequence::new(vec!["intro".to_string(), "simp".to_string()]);
        assert_eq!(seq.len(), 2);
        assert!(!seq.is_empty());
    }

    #[test]
    fn test_tactic_sequence_from_slice() {
        let seq = TacticSequence::from_slice(&["intro", "simp", "decide"]);
        assert_eq!(seq.tactics, vec!["intro", "simp", "decide"]);
    }

    #[test]
    fn test_tactic_sequence_is_prefix() {
        let prefix = TacticSequence::from_slice(&["intro", "simp"]);
        let full = TacticSequence::from_slice(&["intro", "simp", "decide"]);
        let other = TacticSequence::from_slice(&["intro", "apply"]);

        assert!(prefix.is_prefix_of(&full));
        assert!(!full.is_prefix_of(&prefix));
        assert!(!other.is_prefix_of(&full));
        assert!(prefix.is_prefix_of(&prefix)); // prefix of itself
    }

    #[test]
    fn test_tactic_sequence_contains() {
        let full = TacticSequence::from_slice(&["intro", "simp", "decide", "trivial"]);
        let sub = TacticSequence::from_slice(&["simp", "decide"]);
        let not_sub = TacticSequence::from_slice(&["simp", "trivial"]);
        let empty = TacticSequence::new(vec![]);

        assert!(full.contains(&sub));
        assert!(!full.contains(&not_sub));
        assert!(full.contains(&empty)); // empty is in everything
        assert!(!sub.contains(&full)); // longer can't be in shorter
    }

    #[test]
    fn test_tactic_sequence_display() {
        let seq = TacticSequence::from_slice(&["intro", "simp", "decide"]);
        assert_eq!(format!("{}", seq), "intro → simp → decide");
    }

    #[test]
    fn test_pattern_stats_success_rate() {
        let mut stats = PatternStats::default();
        assert_eq!(stats.success_rate(), 0.0);

        stats.success_count = 8;
        stats.failure_count = 2;
        assert!((stats.success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_pattern_stats_wilson_score() {
        let stats = PatternStats {
            success_count: 10,
            failure_count: 0,
            ..Default::default()
        };
        let score = stats.wilson_score();
        // 100% success with n=10 should give Wilson score around 0.72
        assert!(score > 0.7 && score < 0.8);
    }

    #[test]
    fn test_proof_pattern_continuations() {
        let mut pattern = ProofPattern::new(TacticSequence::from_slice(&["intro"]));
        pattern.continuations.insert("simp".to_string(), 5);
        pattern.continuations.insert("apply".to_string(), 3);

        let best = pattern.best_continuation().unwrap();
        assert_eq!(best.0, "simp");
        assert_eq!(best.1, 5);

        let top2 = pattern.top_continuations(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, "simp");
        assert_eq!(top2[1].0, "apply");
    }

    #[test]
    fn test_pattern_extractor_basic() {
        let mut extractor = PatternExtractor::new(3);

        extractor.record_sequence(&["intro", "simp", "decide"]);
        extractor.record_sequence(&["intro", "simp", "trivial"]);
        extractor.record_sequence(&["intro", "apply", "exact"]);

        assert_eq!(extractor.sequence_count(), 3);

        let patterns = extractor.extract_patterns(2);

        // "intro" should appear 3 times
        let intro_pattern = patterns
            .iter()
            .find(|p| p.sequence.tactics == vec!["intro"]);
        assert!(intro_pattern.is_some());
        assert_eq!(intro_pattern.unwrap().stats.success_count, 3);

        // "intro, simp" should appear 2 times
        let intro_simp = patterns
            .iter()
            .find(|p| p.sequence.tactics == vec!["intro", "simp"]);
        assert!(intro_simp.is_some());
        assert_eq!(intro_simp.unwrap().stats.success_count, 2);
    }

    #[test]
    fn test_pattern_extractor_continuations() {
        let mut extractor = PatternExtractor::new(3);

        extractor.record_sequence(&["intro", "simp", "decide"]);
        extractor.record_sequence(&["intro", "simp", "trivial"]);

        let patterns = extractor.extract_patterns(1);

        // Find the "intro, simp" pattern
        let intro_simp = patterns
            .iter()
            .find(|p| p.sequence.tactics == vec!["intro", "simp"])
            .unwrap();

        // Should have continuations "decide" and "trivial"
        assert_eq!(intro_simp.continuations.len(), 2);
        assert!(intro_simp.continuations.contains_key("decide"));
        assert!(intro_simp.continuations.contains_key("trivial"));
    }

    #[test]
    fn test_pattern_database_suggest_continuations() {
        let mut extractor = PatternExtractor::new(4);

        extractor.record_sequence(&["intro", "simp", "decide", "done"]);
        extractor.record_sequence(&["intro", "simp", "trivial", "done"]);
        extractor.record_sequence(&["intro", "apply", "exact", "done"]);

        let patterns = extractor.extract_patterns(1);
        let db = PatternDatabase::from_patterns(patterns);

        // Given we've done "intro, simp", suggest continuations
        let suggestions = db.suggest_continuations(&["intro", "simp"], 5);

        // Should suggest "decide, done" and "trivial, done"
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_pattern_database_persistence() {
        let patterns = vec![ProofPattern::new(TacticSequence::from_slice(&[
            "intro", "simp",
        ]))];
        let db = PatternDatabase::from_patterns(patterns);

        let temp_file =
            std::env::temp_dir().join(format!("pattern_db_test_{}.json", std::process::id()));
        db.save_to_file(&temp_file).unwrap();

        let loaded = PatternDatabase::load_from_file(&temp_file).unwrap();
        assert_eq!(loaded.len(), 1);

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_proof_repairer_suggest_repairs() {
        let mut extractor = PatternExtractor::new(4);

        extractor.record_sequence(&["intro", "simp", "decide"]);
        extractor.record_sequence(&["intro", "simp", "decide"]);
        extractor.record_sequence(&["intro", "apply", "exact"]);

        let patterns = extractor.extract_patterns(1);
        let db = PatternDatabase::from_patterns(patterns);
        let repairer = ProofRepairer::new(db);

        let context = TacticContext {
            property_type: "theorem".to_string(),
            has_quantifiers: false,
            has_implications: false,
            has_arithmetic: false,
            complexity: ContextComplexity::Low,
        };

        // User tried "intro, simp" but failed - suggest repairs
        let repairs = repairer.suggest_repairs(&["intro", "simp"], &context);

        // Should suggest "decide" as continuation
        assert!(!repairs.is_empty());
    }

    #[test]
    fn test_repair_strategy_display() {
        assert_eq!(format!("{}", RepairStrategy::Continue), "continue");
        assert_eq!(format!("{}", RepairStrategy::Backtrack(2)), "backtrack 2");
        assert_eq!(format!("{}", RepairStrategy::Restart), "restart");
    }

    #[test]
    fn test_suggested_continuation_display() {
        let suggestion = SuggestedContinuation {
            tactics: vec!["simp".to_string(), "decide".to_string()],
            pattern_name: Some("simple_proof".to_string()),
            confidence: 0.85,
            pattern_frequency: 10,
        };
        let display = format!("{}", suggestion);
        assert!(display.contains("simp → decide"));
        assert!(display.contains("simple_proof"));
        assert!(display.contains("85.0%"));
    }

    #[test]
    fn test_repair_suggestion_display() {
        let suggestion = RepairSuggestion {
            strategy: RepairStrategy::Continue,
            tactics: vec!["decide".to_string()],
            confidence: 0.75,
            explanation: "Continue with common pattern".to_string(),
        };
        let display = format!("{}", suggestion);
        assert!(display.contains("[continue]"));
        assert!(display.contains("decide"));
        assert!(display.contains("75.0%"));
    }

    #[test]
    fn test_pattern_extractor_clear() {
        let mut extractor = PatternExtractor::new(3);
        extractor.record_sequence(&["intro", "simp"]);

        assert_eq!(extractor.sequence_count(), 1);
        assert!(extractor.ngram_count() > 0);

        extractor.clear();

        assert_eq!(extractor.sequence_count(), 0);
        assert_eq!(extractor.ngram_count(), 0);
    }

    #[test]
    fn test_pattern_database_empty() {
        let db = PatternDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_avg_position_computation() {
        let mut extractor = PatternExtractor::new(3);

        // "simp" appears at position 1 (middle) in both sequences
        extractor.record_sequence(&["intro", "simp", "decide"]); // pos 1/2 = 0.5
        extractor.record_sequence(&["intro", "simp", "trivial"]); // pos 1/2 = 0.5

        let patterns = extractor.extract_patterns(1);

        let simp_pattern = patterns
            .iter()
            .find(|p| p.sequence.tactics == vec!["simp"])
            .unwrap();

        // Average position should be 0.5
        assert!((simp_pattern.stats.avg_position - 0.5).abs() < 0.001);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify TacticSequence::len returns correct length
    #[kani::proof]
    fn verify_sequence_len() {
        let len: usize = kani::any();
        kani::assume(len <= 10);

        let tactics: Vec<String> = (0..len).map(|i| format!("tactic_{}", i)).collect();
        let seq = TacticSequence::new(tactics);

        assert_eq!(seq.len(), len);
    }

    /// Verify is_empty matches len() == 0
    #[kani::proof]
    fn verify_sequence_is_empty() {
        let empty = TacticSequence::new(vec![]);
        let non_empty = TacticSequence::new(vec!["intro".to_string()]);

        assert!(empty.is_empty());
        assert!(!non_empty.is_empty());
        assert_eq!(empty.is_empty(), empty.len() == 0);
        assert_eq!(non_empty.is_empty(), non_empty.len() == 0);
    }

    /// Verify prefix relationship is reflexive
    #[kani::proof]
    fn verify_prefix_reflexive() {
        let seq = TacticSequence::new(vec!["intro".to_string(), "simp".to_string()]);
        assert!(seq.is_prefix_of(&seq));
    }

    /// Verify empty sequence is prefix of everything
    #[kani::proof]
    fn verify_empty_prefix() {
        let empty = TacticSequence::new(vec![]);
        let other = TacticSequence::new(vec!["intro".to_string()]);

        assert!(empty.is_prefix_of(&other));
        assert!(empty.is_prefix_of(&empty));
    }

    /// Verify contains is reflexive
    #[kani::proof]
    fn verify_contains_reflexive() {
        let seq = TacticSequence::new(vec!["intro".to_string(), "simp".to_string()]);
        assert!(seq.contains(&seq));
    }

    /// Verify empty is contained in everything
    #[kani::proof]
    fn verify_empty_contained() {
        let empty = TacticSequence::new(vec![]);
        let other = TacticSequence::new(vec!["intro".to_string()]);

        assert!(other.contains(&empty));
        assert!(empty.contains(&empty));
    }

    /// Verify PatternStats::success_rate is bounded [0, 1]
    #[kani::proof]
    fn verify_success_rate_bounded() {
        let success: u32 = kani::any();
        let failure: u32 = kani::any();

        kani::assume(success <= 1000);
        kani::assume(failure <= 1000);

        let stats = PatternStats {
            success_count: success,
            failure_count: failure,
            ..Default::default()
        };

        let rate = stats.success_rate();
        assert!(rate >= 0.0);
        assert!(rate <= 1.0);
    }

    /// Verify PatternStats::total equals sum of counts
    #[kani::proof]
    fn verify_total_sum() {
        let success: u32 = kani::any();
        let failure: u32 = kani::any();

        kani::assume(success <= 1000);
        kani::assume(failure <= 1000);

        let stats = PatternStats {
            success_count: success,
            failure_count: failure,
            ..Default::default()
        };

        assert_eq!(stats.total(), success + failure);
    }

    /// Verify wilson_score returns 0 for empty stats
    #[kani::proof]
    fn verify_wilson_zero_on_empty() {
        let stats = PatternStats::default();
        assert!(stats.wilson_score() == 0.0);
    }
}
