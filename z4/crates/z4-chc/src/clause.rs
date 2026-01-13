//! Horn clause definitions

use crate::{ChcExpr, PredicateId};
use std::fmt;

/// Body of a Horn clause
///
/// The body is a conjunction of:
/// - Predicate applications (uninterpreted relation calls)
/// - Constraint formula (interpreted theory formula)
#[derive(Debug, Clone)]
pub struct ClauseBody {
    /// Predicate applications: (predicate_id, arguments)
    pub predicates: Vec<(PredicateId, Vec<ChcExpr>)>,
    /// Constraint (background theory formula)
    pub constraint: Option<ChcExpr>,
}

impl ClauseBody {
    /// Create a body with predicates and a constraint
    pub fn new(predicates: Vec<(PredicateId, Vec<ChcExpr>)>, constraint: Option<ChcExpr>) -> Self {
        Self {
            predicates,
            constraint,
        }
    }

    /// Create a body with only a constraint (no predicate applications)
    pub fn constraint(c: ChcExpr) -> Self {
        Self {
            predicates: Vec::new(),
            constraint: Some(c),
        }
    }

    /// Create a body with only predicate applications
    pub fn predicates_only(predicates: Vec<(PredicateId, Vec<ChcExpr>)>) -> Self {
        Self {
            predicates,
            constraint: None,
        }
    }

    /// Create an empty body (represents "true")
    pub fn empty() -> Self {
        Self {
            predicates: Vec::new(),
            constraint: None,
        }
    }

    /// Check if the body is empty (true)
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty() && self.constraint.is_none()
    }

    /// Check if this is a fact (no predicates in body)
    pub fn is_fact(&self) -> bool {
        self.predicates.is_empty()
    }

    /// Get all variables in the body
    pub fn vars(&self) -> Vec<crate::ChcVar> {
        let mut result = Vec::new();
        for (_, args) in &self.predicates {
            for arg in args {
                for v in arg.vars() {
                    if !result.contains(&v) {
                        result.push(v);
                    }
                }
            }
        }
        if let Some(c) = &self.constraint {
            for v in c.vars() {
                if !result.contains(&v) {
                    result.push(v);
                }
            }
        }
        result
    }
}

impl fmt::Display for ClauseBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        for (pred, args) in &self.predicates {
            let args_str: Vec<_> = args.iter().map(|a| a.to_string()).collect();
            parts.push(format!("{}({})", pred, args_str.join(", ")));
        }
        if let Some(c) = &self.constraint {
            parts.push(c.to_string());
        }
        if parts.is_empty() {
            write!(f, "true")
        } else {
            write!(f, "{}", parts.join(" /\\ "))
        }
    }
}

/// Head of a Horn clause
#[derive(Debug, Clone)]
pub enum ClauseHead {
    /// Predicate application
    Predicate(PredicateId, Vec<ChcExpr>),
    /// False (used for queries/safety properties)
    False,
}

impl ClauseHead {
    /// Check if this is a query (head is false)
    pub fn is_query(&self) -> bool {
        matches!(self, ClauseHead::False)
    }

    /// Get the predicate ID if this is a predicate head
    pub fn predicate_id(&self) -> Option<PredicateId> {
        match self {
            ClauseHead::Predicate(id, _) => Some(*id),
            ClauseHead::False => None,
        }
    }

    /// Get all variables in the head
    pub fn vars(&self) -> Vec<crate::ChcVar> {
        match self {
            ClauseHead::Predicate(_, args) => {
                let mut result = Vec::new();
                for arg in args {
                    for v in arg.vars() {
                        if !result.contains(&v) {
                            result.push(v);
                        }
                    }
                }
                result
            }
            ClauseHead::False => Vec::new(),
        }
    }
}

impl fmt::Display for ClauseHead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClauseHead::Predicate(pred, args) => {
                let args_str: Vec<_> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{}({})", pred, args_str.join(", "))
            }
            ClauseHead::False => write!(f, "false"),
        }
    }
}

/// A Constrained Horn Clause
///
/// Has the form: `forall vars. body => head`
/// where body is a conjunction of predicate applications and constraints,
/// and head is either a predicate application or false.
#[derive(Debug, Clone)]
pub struct HornClause {
    pub body: ClauseBody,
    pub head: ClauseHead,
}

impl HornClause {
    /// Create a new Horn clause
    pub fn new(body: ClauseBody, head: ClauseHead) -> Self {
        Self { body, head }
    }

    /// Create a fact: constraint => P(args)
    pub fn fact(constraint: ChcExpr, pred: PredicateId, args: Vec<ChcExpr>) -> Self {
        Self {
            body: ClauseBody::constraint(constraint),
            head: ClauseHead::Predicate(pred, args),
        }
    }

    /// Create a query: body => false
    pub fn query(body: ClauseBody) -> Self {
        Self {
            body,
            head: ClauseHead::False,
        }
    }

    /// Check if this is a query clause
    pub fn is_query(&self) -> bool {
        self.head.is_query()
    }

    /// Check if this is a fact (no predicates in body)
    pub fn is_fact(&self) -> bool {
        self.body.is_fact()
    }

    /// Get all variables in the clause
    pub fn vars(&self) -> Vec<crate::ChcVar> {
        let mut result = self.body.vars();
        for v in self.head.vars() {
            if !result.contains(&v) {
                result.push(v);
            }
        }
        result
    }

    /// Get all predicate IDs used in the clause
    pub fn predicate_ids(&self) -> Vec<PredicateId> {
        let mut result: Vec<_> = self.body.predicates.iter().map(|(id, _)| *id).collect();
        if let Some(id) = self.head.predicate_id() {
            if !result.contains(&id) {
                result.push(id);
            }
        }
        result
    }
}

impl fmt::Display for HornClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} => {}", self.body, self.head)
    }
}
