//! Premise Selection for Automated Theorem Proving
//!
//! Implements two complementary selection strategies from Isabelle:
//!
//! ## MePo (Meng-Paulson) - Symbol-Based Relevance
//! Ranks premises by symbol overlap with the goal. Rare symbols get higher weight.
//! Fast, interpretable, and effective for goals with distinctive constants.
//!
//! ## MaSh (Machine-learning for Sledgehammer) - Feature-Based Learning
//! Extracts features from terms and uses k-NN / Naive Bayes to predict
//! which premises are likely useful based on past proof attempts.
//!
//! # References
//! - "Lightweight Relevance Filtering for Machine-Generated Resolution Problems" (Meng & Paulson, 2009)
//! - "MaSh: Machine Learning for Sledgehammer" (Kühlwein et al., 2013)

use lean5_kernel::{Expr, Name};
use std::collections::{HashMap, HashSet};

/// A feature extracted from an expression for ML-based premise selection
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Feature {
    /// Constant symbol (function/type name)
    Const(Name),
    /// Type constructor
    Type(Name),
    /// Application pattern: f applied to something
    App(Name),
    /// Binary application pattern: (f _ _)
    BinApp(Name),
    /// Theory marker (e.g., "arith", "set", "list")
    Theory(String),
    /// Depth-limited subterm pattern
    Pattern(String),
}

impl Feature {
    /// Create a constant feature
    pub fn constant(name: Name) -> Self {
        Feature::Const(name)
    }

    /// Create a type feature
    pub fn type_const(name: Name) -> Self {
        Feature::Type(name)
    }

    /// Create an application feature
    pub fn application(name: Name) -> Self {
        Feature::App(name)
    }
}

/// Feature set for a term/goal
#[derive(Clone, Debug, Default)]
pub struct FeatureSet {
    features: HashSet<Feature>,
}

impl FeatureSet {
    /// Create a new empty feature set
    pub fn new() -> Self {
        Self {
            features: HashSet::new(),
        }
    }

    /// Add a feature
    pub fn add(&mut self, f: Feature) {
        self.features.insert(f);
    }

    /// Get all features
    pub fn features(&self) -> &HashSet<Feature> {
        &self.features
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Number of features
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Compute overlap (intersection size) with another feature set
    pub fn overlap(&self, other: &FeatureSet) -> usize {
        self.features.intersection(&other.features).count()
    }

    /// Compute Jaccard similarity with another feature set
    pub fn jaccard(&self, other: &FeatureSet) -> f64 {
        let intersection = self.overlap(other);
        let union = self.features.union(&other.features).count();
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

/// Extract features from an expression
pub struct FeatureExtractor {
    /// Maximum depth for recursive feature extraction
    max_depth: usize,
    /// Whether to include type features
    include_types: bool,
    /// Whether to include application patterns
    include_patterns: bool,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self {
            max_depth: 3,
            include_types: true,
            include_patterns: true,
        }
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum extraction depth
    #[must_use]
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Enable/disable type features
    #[must_use]
    pub fn with_types(mut self, include: bool) -> Self {
        self.include_types = include;
        self
    }

    /// Enable/disable pattern features
    #[must_use]
    pub fn with_patterns(mut self, include: bool) -> Self {
        self.include_patterns = include;
        self
    }

    /// Extract features from an expression
    pub fn extract(&self, expr: &Expr) -> FeatureSet {
        let mut features = FeatureSet::new();
        self.extract_recursive(expr, 0, &mut features);
        features
    }

    fn extract_recursive(&self, expr: &Expr, depth: usize, features: &mut FeatureSet) {
        if depth > self.max_depth {
            return;
        }

        match expr {
            Expr::Const(name, _levels) => {
                features.add(Feature::Const(name.clone()));
                // Add theory feature based on name prefix
                if let Some(theory) = self.detect_theory(name) {
                    features.add(Feature::Theory(theory));
                }
            }

            Expr::App(f, arg) => {
                // Extract from both parts
                self.extract_recursive(f, depth + 1, features);
                self.extract_recursive(arg, depth + 1, features);

                // Add application pattern feature
                if self.include_patterns {
                    if let Expr::Const(name, _) = f.as_ref() {
                        features.add(Feature::App(name.clone()));
                    }
                    // Check for binary application
                    if let Expr::App(ff, _) = f.as_ref() {
                        if let Expr::Const(name, _) = ff.as_ref() {
                            features.add(Feature::BinApp(name.clone()));
                        }
                    }
                }
            }

            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                if self.include_types {
                    self.extract_recursive(ty, depth + 1, features);
                }
                self.extract_recursive(body, depth + 1, features);
            }

            Expr::Let(ty, val, body) => {
                if self.include_types {
                    self.extract_recursive(ty, depth + 1, features);
                }
                self.extract_recursive(val, depth + 1, features);
                self.extract_recursive(body, depth + 1, features);
            }

            Expr::Proj(name, _idx, struct_expr) => {
                features.add(Feature::Const(name.clone()));
                self.extract_recursive(struct_expr, depth + 1, features);
            }

            Expr::Sort(_) | Expr::BVar(_) | Expr::FVar(_) | Expr::Lit(_) => {
                // Terminal expressions - no features
            }

            // MData is transparent - extract from inner expression
            Expr::MData(_, inner) => {
                self.extract_recursive(inner, depth, features);
            }

            // Mode-specific expressions - no features for now
            Expr::CubicalInterval
            | Expr::CubicalI0
            | Expr::CubicalI1
            | Expr::CubicalPath { .. }
            | Expr::CubicalPathLam { .. }
            | Expr::CubicalPathApp { .. }
            | Expr::CubicalHComp { .. }
            | Expr::CubicalTransp { .. }
            | Expr::ClassicalChoice { .. }
            | Expr::ClassicalEpsilon { .. }
            | Expr::ZFCSet(_)
            | Expr::ZFCMem { .. }
            | Expr::ZFCComprehension { .. }
            | Expr::SProp
            | Expr::Squash(_) => {
                // Mode-specific expressions - no feature extraction yet
            }
        }
    }

    /// Detect theory based on constant name
    fn detect_theory(&self, name: &Name) -> Option<String> {
        let s = name.to_string();
        // Common prefixes for different theories
        if s.starts_with("Nat.") || s.starts_with("Int.") {
            Some("arith".to_string())
        } else if s.starts_with("List.") || s.starts_with("Array.") {
            Some("list".to_string())
        } else if s.starts_with("Set.") || s.starts_with("Finset.") {
            Some("set".to_string())
        } else if s.starts_with("Real.") || s.starts_with("Complex.") {
            Some("analysis".to_string())
        } else if s.starts_with("String.") || s.starts_with("Char.") {
            Some("string".to_string())
        } else {
            None
        }
    }

    /// Extract all constants from an expression (for MePo)
    pub fn extract_constants(&self, expr: &Expr) -> HashSet<Name> {
        let mut constants = HashSet::new();
        self.extract_constants_recursive(expr, &mut constants);
        constants
    }

    fn extract_constants_recursive(&self, expr: &Expr, constants: &mut HashSet<Name>) {
        match expr {
            Expr::Const(name, _) => {
                constants.insert(name.clone());
            }
            Expr::App(f, arg) => {
                self.extract_constants_recursive(f, constants);
                self.extract_constants_recursive(arg, constants);
            }
            Expr::Lam(_, ty, body) | Expr::Pi(_, ty, body) => {
                self.extract_constants_recursive(ty, constants);
                self.extract_constants_recursive(body, constants);
            }
            Expr::Let(ty, val, body) => {
                self.extract_constants_recursive(ty, constants);
                self.extract_constants_recursive(val, constants);
                self.extract_constants_recursive(body, constants);
            }
            Expr::Proj(name, _, struct_expr) => {
                constants.insert(name.clone());
                self.extract_constants_recursive(struct_expr, constants);
            }
            Expr::Sort(_) | Expr::BVar(_) | Expr::FVar(_) | Expr::Lit(_) => {}
            // MData is transparent - extract constants from inner
            Expr::MData(_, inner) => {
                self.extract_constants_recursive(inner, constants);
            }
            // Mode-specific expressions - no constants
            Expr::CubicalInterval
            | Expr::CubicalI0
            | Expr::CubicalI1
            | Expr::CubicalPath { .. }
            | Expr::CubicalPathLam { .. }
            | Expr::CubicalPathApp { .. }
            | Expr::CubicalHComp { .. }
            | Expr::CubicalTransp { .. }
            | Expr::ClassicalChoice { .. }
            | Expr::ClassicalEpsilon { .. }
            | Expr::ZFCSet(_)
            | Expr::ZFCMem { .. }
            | Expr::ZFCComprehension { .. }
            | Expr::SProp
            | Expr::Squash(_) => {}
        }
    }
}

/// A known fact/premise in the database
#[derive(Clone, Debug)]
pub struct Premise {
    /// Unique identifier for this premise
    pub id: PremiseId,
    /// Name of the theorem/lemma
    pub name: Name,
    /// The statement (type of the theorem)
    pub statement: Expr,
    /// Extracted features for ML-based selection
    pub features: FeatureSet,
    /// Constants appearing in this premise (for MePo)
    pub constants: HashSet<Name>,
    /// Dependencies (other premises used in the proof)
    pub dependencies: Vec<PremiseId>,
}

/// Unique identifier for a premise
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PremiseId(pub u64);

impl Premise {
    /// Create a new premise
    pub fn new(id: PremiseId, name: Name, statement: Expr) -> Self {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract(&statement);
        let constants = extractor.extract_constants(&statement);
        Self {
            id,
            name,
            statement,
            features,
            constants,
            dependencies: Vec::new(),
        }
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, dep: PremiseId) {
        if !self.dependencies.contains(&dep) {
            self.dependencies.push(dep);
        }
    }
}

/// Database of known premises
#[derive(Default)]
pub struct PremiseDatabase {
    /// All premises indexed by ID
    premises: HashMap<PremiseId, Premise>,
    /// Premises indexed by name
    by_name: HashMap<Name, PremiseId>,
    /// Global constant frequencies (for MePo weighting)
    const_freq: HashMap<Name, usize>,
    /// Total number of premises
    count: u64,
    /// Next available premise ID
    next_id: u64,
}

impl PremiseDatabase {
    /// Create a new empty premise database
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a premise to the database
    pub fn add(&mut self, name: Name, statement: Expr) -> PremiseId {
        let id = PremiseId(self.next_id);
        self.next_id += 1;
        self.count += 1;

        let premise = Premise::new(id, name.clone(), statement);

        // Update constant frequencies
        for c in &premise.constants {
            *self.const_freq.entry(c.clone()).or_insert(0) += 1;
        }

        self.by_name.insert(name, id);
        self.premises.insert(id, premise);
        id
    }

    /// Get a premise by ID
    pub fn get(&self, id: PremiseId) -> Option<&Premise> {
        self.premises.get(&id)
    }

    /// Get a premise by name
    pub fn get_by_name(&self, name: &Name) -> Option<&Premise> {
        self.by_name.get(name).and_then(|id| self.premises.get(id))
    }

    /// Get the frequency of a constant
    pub fn const_frequency(&self, name: &Name) -> usize {
        self.const_freq.get(name).copied().unwrap_or(0)
    }

    /// Total number of premises
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Iterate over all premises
    pub fn iter(&self) -> impl Iterator<Item = &Premise> {
        self.premises.values()
    }

    /// Record a successful proof: update dependencies
    pub fn record_proof(&mut self, proved: PremiseId, used_premises: &[PremiseId]) {
        if let Some(premise) = self.premises.get_mut(&proved) {
            for &dep in used_premises {
                premise.add_dependency(dep);
            }
        }
    }
}

/// MePo (Meng-Paulson) Symbol-Based Premise Selection
///
/// Ranks premises by weighted symbol overlap with the goal.
/// Rare symbols receive higher weight using the formula:
///   weight(c) = 1 + 2 / ln(freq(c) + 1)
pub struct MePoSelector<'a> {
    db: &'a PremiseDatabase,
    /// Relevance threshold (0.0 to 1.0)
    threshold: f64,
    /// Maximum number of premises to select
    max_premises: usize,
}

impl<'a> MePoSelector<'a> {
    /// Create a new MePo selector
    pub fn new(db: &'a PremiseDatabase) -> Self {
        Self {
            db,
            threshold: 0.1,
            max_premises: 64,
        }
    }

    /// Set the relevance threshold
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the maximum number of premises
    #[must_use]
    pub fn with_max_premises(mut self, max: usize) -> Self {
        self.max_premises = max;
        self
    }

    /// Compute the weight of a constant based on its rarity
    fn const_weight(&self, name: &Name) -> f64 {
        let freq = self.db.const_frequency(name);
        1.0 + 2.0 / (freq as f64 + 1.0).ln()
    }

    /// Compute relevance score for a premise given goal constants
    fn relevance(&self, premise: &Premise, goal_constants: &HashSet<Name>) -> f64 {
        let mut score = 0.0;
        let mut max_possible = 0.0;

        // Compute weighted overlap
        for c in goal_constants {
            let w = self.const_weight(c);
            max_possible += w;
            if premise.constants.contains(c) {
                score += w;
            }
        }

        // Normalize by maximum possible score
        if max_possible > 0.0 {
            score / max_possible
        } else {
            0.0
        }
    }

    /// Select relevant premises for a goal
    pub fn select(&self, goal: &Expr) -> Vec<&Premise> {
        let extractor = FeatureExtractor::new();
        let goal_constants = extractor.extract_constants(goal);

        if goal_constants.is_empty() {
            return Vec::new();
        }

        // Score all premises
        let mut scored: Vec<_> = self
            .db
            .iter()
            .map(|p| {
                let score = self.relevance(p, &goal_constants);
                (p, score)
            })
            .filter(|(_, score)| *score >= self.threshold)
            .collect();

        // Sort by score (descending), then by premise ID for stability
        scored.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
            Some(std::cmp::Ordering::Equal) | None => a.0.id.0.cmp(&b.0.id.0),
            Some(ord) => ord,
        });

        // Take top N
        scored
            .into_iter()
            .take(self.max_premises)
            .map(|(p, _)| p)
            .collect()
    }

    /// Select with scores (for debugging/analysis)
    pub fn select_with_scores(&self, goal: &Expr) -> Vec<(&Premise, f64)> {
        let extractor = FeatureExtractor::new();
        let goal_constants = extractor.extract_constants(goal);

        if goal_constants.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<_> = self
            .db
            .iter()
            .map(|p| {
                let score = self.relevance(p, &goal_constants);
                (p, score)
            })
            .filter(|(_, score)| *score >= self.threshold)
            .collect();

        // Sort by score (descending), then by premise ID for stability
        scored.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
            Some(std::cmp::Ordering::Equal) | None => a.0.id.0.cmp(&b.0.id.0),
            Some(ord) => ord,
        });
        scored.into_iter().take(self.max_premises).collect()
    }
}

/// MaSh (Machine Learning for Sledgehammer) Feature-Based Premise Selection
///
/// Uses k-NN and/or Naive Bayes to predict useful premises based on
/// feature similarity and learning from past proof attempts.
pub struct MaShSelector<'a> {
    db: &'a PremiseDatabase,
    /// Proof history: goal features -> useful premises
    proof_history: Vec<ProofRecord>,
    /// k for k-NN
    k: usize,
    /// Maximum premises to return
    max_premises: usize,
    /// Use naive Bayes in addition to k-NN
    use_naive_bayes: bool,
}

/// Record of a successful proof for learning
#[derive(Clone, Debug)]
pub struct ProofRecord {
    /// Features of the proved goal
    pub goal_features: FeatureSet,
    /// Premises that were useful
    pub useful_premises: Vec<PremiseId>,
}

impl<'a> MaShSelector<'a> {
    /// Create a new MaSh selector
    pub fn new(db: &'a PremiseDatabase) -> Self {
        Self {
            db,
            proof_history: Vec::new(),
            k: 16,
            max_premises: 64,
            use_naive_bayes: true,
        }
    }

    /// Set k for k-NN
    #[must_use]
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set maximum premises
    #[must_use]
    pub fn with_max_premises(mut self, max: usize) -> Self {
        self.max_premises = max;
        self
    }

    /// Enable/disable Naive Bayes
    #[must_use]
    pub fn with_naive_bayes(mut self, use_nb: bool) -> Self {
        self.use_naive_bayes = use_nb;
        self
    }

    /// Record a successful proof for learning
    pub fn record_proof(&mut self, goal: &Expr, useful_premises: Vec<PremiseId>) {
        let extractor = FeatureExtractor::new();
        let goal_features = extractor.extract(goal);
        self.proof_history.push(ProofRecord {
            goal_features,
            useful_premises,
        });
    }

    /// Select premises using k-NN
    fn select_knn(&self, goal_features: &FeatureSet) -> HashMap<PremiseId, f64> {
        let mut premise_scores: HashMap<PremiseId, f64> = HashMap::new();

        if self.proof_history.is_empty() {
            return premise_scores;
        }

        // Find k nearest neighbors by feature similarity
        let mut neighbors: Vec<_> = self
            .proof_history
            .iter()
            .map(|record| {
                let sim = goal_features.jaccard(&record.goal_features);
                (record, sim)
            })
            .collect();

        neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k_nearest: Vec<_> = neighbors.into_iter().take(self.k).collect();

        // Aggregate premise scores from neighbors
        for (record, sim) in k_nearest {
            for &premise_id in &record.useful_premises {
                *premise_scores.entry(premise_id).or_insert(0.0) += sim;
            }
        }

        premise_scores
    }

    /// Select premises using Naive Bayes
    fn select_naive_bayes(&self, goal_features: &FeatureSet) -> HashMap<PremiseId, f64> {
        let mut premise_scores: HashMap<PremiseId, f64> = HashMap::new();

        // Compute feature -> premise associations from history
        let mut feature_premise_count: HashMap<&Feature, HashMap<PremiseId, usize>> =
            HashMap::new();
        let mut feature_count: HashMap<&Feature, usize> = HashMap::new();
        let mut premise_count: HashMap<PremiseId, usize> = HashMap::new();

        for record in &self.proof_history {
            for feature in record.goal_features.features() {
                *feature_count.entry(feature).or_insert(0) += 1;
                for &premise_id in &record.useful_premises {
                    *feature_premise_count
                        .entry(feature)
                        .or_default()
                        .entry(premise_id)
                        .or_insert(0) += 1;
                }
            }
            for &premise_id in &record.useful_premises {
                *premise_count.entry(premise_id).or_insert(0) += 1;
            }
        }

        let total_records = self.proof_history.len() as f64;
        if total_records == 0.0 {
            return premise_scores;
        }

        // For each premise, compute P(premise | goal_features) using Naive Bayes
        for premise in self.db.iter() {
            let prior = (*premise_count.get(&premise.id).unwrap_or(&0) as f64 + 1.0)
                / (total_records + 2.0);

            let mut log_likelihood = prior.ln();

            for feature in goal_features.features() {
                if let Some(fp_count) = feature_premise_count.get(feature) {
                    let count = *fp_count.get(&premise.id).unwrap_or(&0) as f64;
                    let feat_count = *feature_count.get(feature).unwrap_or(&0) as f64;
                    // Laplace smoothing
                    let prob = (count + 1.0) / (feat_count + 2.0);
                    log_likelihood += prob.ln();
                }
            }

            if log_likelihood.is_finite() {
                premise_scores.insert(premise.id, log_likelihood);
            }
        }

        premise_scores
    }

    /// Select premises for a goal
    pub fn select(&self, goal: &Expr) -> Vec<&Premise> {
        let extractor = FeatureExtractor::new();
        let goal_features = extractor.extract(goal);

        // Combine k-NN and Naive Bayes scores
        let knn_scores = self.select_knn(&goal_features);

        let mut combined_scores: HashMap<PremiseId, f64> = knn_scores;

        if self.use_naive_bayes {
            let nb_scores = self.select_naive_bayes(&goal_features);

            // Normalize Naive Bayes scores to [0, 1]
            let nb_max = nb_scores
                .values()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let nb_min = nb_scores.values().copied().fold(f64::INFINITY, f64::min);
            let nb_range = nb_max - nb_min;

            for (id, score) in nb_scores {
                let normalized = if nb_range > 0.0 {
                    (score - nb_min) / nb_range
                } else {
                    0.5
                };
                *combined_scores.entry(id).or_insert(0.0) += normalized;
            }
        }

        // Fallback: if no proof history, use feature similarity to premises
        if combined_scores.is_empty() {
            for premise in self.db.iter() {
                let sim = goal_features.jaccard(&premise.features);
                if sim > 0.0 {
                    combined_scores.insert(premise.id, sim);
                }
            }
        }

        // Sort and return top premises
        let mut scored: Vec<_> = combined_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(self.max_premises)
            .filter_map(|(id, _)| self.db.get(id))
            .collect()
    }
}

/// Combined premise selector using both MePo and MaSh
pub struct HybridSelector<'a> {
    db: &'a PremiseDatabase,
    /// Weight for MePo scores (0.0 to 1.0)
    mepo_weight: f64,
    /// Weight for MaSh scores (0.0 to 1.0)
    mash_weight: f64,
    /// Maximum premises to return
    max_premises: usize,
    /// Proof history for MaSh
    proof_history: Vec<ProofRecord>,
}

impl<'a> HybridSelector<'a> {
    /// Create a new hybrid selector
    pub fn new(db: &'a PremiseDatabase) -> Self {
        Self {
            db,
            mepo_weight: 0.5,
            mash_weight: 0.5,
            max_premises: 64,
            proof_history: Vec::new(),
        }
    }

    /// Set MePo weight
    #[must_use]
    pub fn with_mepo_weight(mut self, weight: f64) -> Self {
        self.mepo_weight = weight;
        self
    }

    /// Set MaSh weight
    #[must_use]
    pub fn with_mash_weight(mut self, weight: f64) -> Self {
        self.mash_weight = weight;
        self
    }

    /// Set maximum premises
    #[must_use]
    pub fn with_max_premises(mut self, max: usize) -> Self {
        self.max_premises = max;
        self
    }

    /// Record a successful proof
    pub fn record_proof(&mut self, goal: &Expr, useful_premises: Vec<PremiseId>) {
        let extractor = FeatureExtractor::new();
        let goal_features = extractor.extract(goal);
        self.proof_history.push(ProofRecord {
            goal_features,
            useful_premises,
        });
    }

    /// Select premises combining MePo and MaSh
    pub fn select(&self, goal: &Expr) -> Vec<&Premise> {
        let extractor = FeatureExtractor::new();
        let goal_features = extractor.extract(goal);
        let goal_constants = extractor.extract_constants(goal);

        let mut combined_scores: HashMap<PremiseId, f64> = HashMap::new();

        // MePo scoring
        if self.mepo_weight > 0.0 {
            let mepo = MePoSelector::new(self.db).with_threshold(0.0);
            for (premise, score) in mepo.select_with_scores(goal) {
                *combined_scores.entry(premise.id).or_insert(0.0) += self.mepo_weight * score;
            }
        }

        // MaSh scoring (k-NN component)
        if self.mash_weight > 0.0 && !self.proof_history.is_empty() {
            // Find similar past goals
            let mut neighbors: Vec<_> = self
                .proof_history
                .iter()
                .map(|record| {
                    let sim = goal_features.jaccard(&record.goal_features);
                    (record, sim)
                })
                .filter(|(_, sim)| *sim > 0.0)
                .collect();

            neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let k = 16.min(neighbors.len());
            for (record, sim) in neighbors.into_iter().take(k) {
                for &premise_id in &record.useful_premises {
                    *combined_scores.entry(premise_id).or_insert(0.0) += self.mash_weight * sim;
                }
            }
        }

        // Fallback: feature similarity to premises
        if combined_scores.is_empty() {
            for premise in self.db.iter() {
                let feat_sim = goal_features.jaccard(&premise.features);
                let const_overlap = premise.constants.intersection(&goal_constants).count() as f64
                    / (goal_constants.len().max(1)) as f64;
                let score = 0.5 * feat_sim + 0.5 * const_overlap;
                if score > 0.0 {
                    combined_scores.insert(premise.id, score);
                }
            }
        }

        // Sort and return
        let mut scored: Vec<_> = combined_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(self.max_premises)
            .filter_map(|(id, _)| self.db.get(id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lean5_kernel::Level;

    fn name(s: &str) -> Name {
        s.parse().unwrap()
    }

    fn make_const(s: &str) -> Expr {
        Expr::const_(name(s), vec![])
    }

    fn make_app(f: &str, arg: Expr) -> Expr {
        Expr::app(make_const(f), arg)
    }

    fn make_app2(f: &str, arg1: Expr, arg2: Expr) -> Expr {
        Expr::app(Expr::app(make_const(f), arg1), arg2)
    }

    #[test]
    fn test_feature_extraction_constant() {
        let expr = make_const("Nat.add");
        let extractor = FeatureExtractor::new();
        let features = extractor.extract(&expr);

        assert!(features
            .features()
            .contains(&Feature::Const(name("Nat.add"))));
        assert!(features
            .features()
            .contains(&Feature::Theory("arith".to_string())));
    }

    #[test]
    fn test_feature_extraction_application() {
        let expr = make_app2("Nat.add", make_const("x"), make_const("y"));
        let extractor = FeatureExtractor::new();
        let features = extractor.extract(&expr);

        assert!(features
            .features()
            .contains(&Feature::Const(name("Nat.add"))));
        assert!(features.features().contains(&Feature::Const(name("x"))));
        assert!(features.features().contains(&Feature::Const(name("y"))));
        assert!(features.features().contains(&Feature::App(name("Nat.add"))));
        assert!(features
            .features()
            .contains(&Feature::BinApp(name("Nat.add"))));
    }

    #[test]
    fn test_feature_extraction_pi() {
        // ∀ x : Nat, P x
        let nat = make_const("Nat");
        let p_x = make_app("P", Expr::bvar(0));
        let expr = Expr::pi(lean5_kernel::BinderInfo::Default, nat, p_x);

        let extractor = FeatureExtractor::new();
        let features = extractor.extract(&expr);

        assert!(features.features().contains(&Feature::Const(name("Nat"))));
        assert!(features.features().contains(&Feature::Const(name("P"))));
    }

    #[test]
    fn test_extract_constants() {
        let expr = make_app2("f", make_const("a"), make_app("g", make_const("b")));
        let extractor = FeatureExtractor::new();
        let constants = extractor.extract_constants(&expr);

        assert!(constants.contains(&name("f")));
        assert!(constants.contains(&name("a")));
        assert!(constants.contains(&name("g")));
        assert!(constants.contains(&name("b")));
        assert_eq!(constants.len(), 4);
    }

    #[test]
    fn test_feature_set_overlap() {
        let mut fs1 = FeatureSet::new();
        fs1.add(Feature::Const(name("a")));
        fs1.add(Feature::Const(name("b")));
        fs1.add(Feature::Const(name("c")));

        let mut fs2 = FeatureSet::new();
        fs2.add(Feature::Const(name("b")));
        fs2.add(Feature::Const(name("c")));
        fs2.add(Feature::Const(name("d")));

        assert_eq!(fs1.overlap(&fs2), 2); // b and c
    }

    #[test]
    fn test_feature_set_jaccard() {
        let mut fs1 = FeatureSet::new();
        fs1.add(Feature::Const(name("a")));
        fs1.add(Feature::Const(name("b")));

        let mut fs2 = FeatureSet::new();
        fs2.add(Feature::Const(name("b")));
        fs2.add(Feature::Const(name("c")));

        // Intersection: {b}, Union: {a, b, c}
        // Jaccard = 1/3
        let jaccard = fs1.jaccard(&fs2);
        assert!((jaccard - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_premise_database() {
        let mut db = PremiseDatabase::new();

        let stmt1 = make_app2("Eq", make_const("a"), make_const("a"));
        let id1 = db.add(name("refl"), stmt1);

        let stmt2 = make_app2("Eq", make_const("a"), make_const("b"));
        let _id2 = db.add(name("hyp"), stmt2);

        assert_eq!(db.len(), 2);
        assert!(db.get(id1).is_some());
        assert!(db.get_by_name(&name("refl")).is_some());

        // Check constant frequencies
        assert_eq!(db.const_frequency(&name("Eq")), 2);
        assert_eq!(db.const_frequency(&name("a")), 2);
        assert_eq!(db.const_frequency(&name("b")), 1);
    }

    #[test]
    fn test_mepo_selection() {
        let mut db = PremiseDatabase::new();

        // Add premises with different symbol profiles
        db.add(
            name("nat_add_comm"),
            make_app2("Nat.add", make_const("x"), make_const("y")),
        );
        db.add(
            name("nat_mul_comm"),
            make_app2("Nat.mul", make_const("x"), make_const("y")),
        );
        db.add(
            name("list_length"),
            make_app("List.length", make_const("xs")),
        );

        // Goal involving Nat.add
        let goal = make_app2("Nat.add", make_const("a"), make_const("b"));

        let selector = MePoSelector::new(&db).with_threshold(0.0);
        let selected = selector.select_with_scores(&goal);

        // nat_add_comm should rank highest (shares Nat.add)
        assert!(!selected.is_empty());
        assert_eq!(selected[0].0.name, name("nat_add_comm"));
    }

    #[test]
    fn test_mepo_rare_symbol_weight() {
        let mut db = PremiseDatabase::new();

        // Add many premises with common symbol
        for i in 0..10 {
            db.add(
                name(&format!("common_{i}")),
                make_app("common_fn", make_const(&format!("x{i}"))),
            );
        }

        // Add one premise with rare symbol
        db.add(name("rare_one"), make_app("rare_fn", make_const("x")));

        // Goal with both rare and common symbols
        let _goal = make_app2("combine", make_const("rare_fn"), make_const("common_fn"));

        let selector = MePoSelector::new(&db);

        // rare_fn weight should be higher than common_fn weight
        let rare_weight = selector.const_weight(&name("rare_fn"));
        let common_weight = selector.const_weight(&name("common_fn"));

        assert!(rare_weight > common_weight);
    }

    #[test]
    fn test_mash_without_history() {
        let mut db = PremiseDatabase::new();

        db.add(name("p1"), make_app("f", make_const("a")));
        db.add(name("p2"), make_app("g", make_const("b")));

        let goal = make_app("f", make_const("x"));

        let selector = MaShSelector::new(&db);
        let selected = selector.select(&goal);

        // Should fall back to feature similarity
        // p1 should rank higher (shares "f")
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_mash_with_history() {
        let mut db = PremiseDatabase::new();

        let id1 = db.add(name("p1"), make_app("f", make_const("a")));
        let id2 = db.add(name("p2"), make_app("g", make_const("b")));
        let _id3 = db.add(name("p3"), make_app("h", make_const("c")));

        let mut selector = MaShSelector::new(&db);

        // Record that p1 and p2 were useful for a goal involving f
        let past_goal = make_app("f", make_const("x"));
        selector.record_proof(&past_goal, vec![id1, id2]);

        // New goal also involving f
        let new_goal = make_app("f", make_const("y"));
        let selected = selector.select(&new_goal);

        // p1 and p2 should be recommended based on history
        let selected_ids: Vec<_> = selected.iter().map(|p| p.id).collect();
        assert!(selected_ids.contains(&id1) || selected_ids.contains(&id2));
    }

    #[test]
    fn test_hybrid_selector() {
        let mut db = PremiseDatabase::new();

        let id1 = db.add(
            name("eq_refl"),
            make_app2("Eq", make_const("x"), make_const("x")),
        );
        let _id2 = db.add(
            name("eq_symm"),
            make_app2("Eq", make_const("y"), make_const("x")),
        );
        let _id3 = db.add(
            name("nat_add"),
            make_app2("Nat.add", make_const("a"), make_const("b")),
        );

        let mut selector = HybridSelector::new(&db)
            .with_mepo_weight(0.6)
            .with_mash_weight(0.4);

        // Record a proof
        let past_goal = make_app2("Eq", make_const("a"), make_const("a"));
        selector.record_proof(&past_goal, vec![id1]);

        // Select for new goal
        let goal = make_app2("Eq", make_const("p"), make_const("q"));
        let selected = selector.select(&goal);

        // eq_refl and eq_symm should rank higher than nat_add
        assert!(!selected.is_empty());
        let top_name = &selected[0].name;
        assert!(
            *top_name == name("eq_refl") || *top_name == name("eq_symm"),
            "Expected eq_refl or eq_symm, got {top_name:?}"
        );
    }

    #[test]
    fn test_theory_detection() {
        let extractor = FeatureExtractor::new();

        let nat_expr = make_const("Nat.succ");
        let features = extractor.extract(&nat_expr);
        assert!(features
            .features()
            .contains(&Feature::Theory("arith".to_string())));

        let list_expr = make_const("List.cons");
        let features = extractor.extract(&list_expr);
        assert!(features
            .features()
            .contains(&Feature::Theory("list".to_string())));

        let set_expr = make_const("Set.union");
        let features = extractor.extract(&set_expr);
        assert!(features
            .features()
            .contains(&Feature::Theory("set".to_string())));
    }

    #[test]
    fn test_max_depth_limiting() {
        // Create a deeply nested expression
        let mut expr = make_const("leaf");
        for i in 0..10 {
            expr = make_app(&format!("f{i}"), expr);
        }

        // With depth 2, should not extract deeply nested features
        let extractor = FeatureExtractor::new().with_depth(2);
        let features = extractor.extract(&expr);

        // Should have some but not all features
        assert!(features.len() < 11);
        assert!(features.features().contains(&Feature::Const(name("f9"))));
    }

    #[test]
    fn test_empty_goal() {
        let db = PremiseDatabase::new();
        let mepo = MePoSelector::new(&db);

        // Goal with only bound variables
        let goal = Expr::lam(
            lean5_kernel::BinderInfo::Default,
            Expr::sort(Level::zero()),
            Expr::bvar(0),
        );

        let selected = mepo.select(&goal);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_premise_dependencies() {
        let mut db = PremiseDatabase::new();

        let id1 = db.add(name("p1"), make_const("a"));
        let id2 = db.add(name("p2"), make_const("b"));
        let id3 = db.add(name("p3"), make_const("c"));

        // Record that p3 was proved using p1 and p2
        db.record_proof(id3, &[id1, id2]);

        let p3 = db.get(id3).unwrap();
        assert_eq!(p3.dependencies.len(), 2);
        assert!(p3.dependencies.contains(&id1));
        assert!(p3.dependencies.contains(&id2));
    }
}
