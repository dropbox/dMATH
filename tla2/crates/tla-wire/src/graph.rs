//! Core wiring graph representation.
//!
//! The wiring graph models an application as nodes (functions, handlers, effects)
//! and edges (calls, data flow, registrations). This is the foundation for
//! reachability analysis and wiring verification.

use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Unique identifier for a node in the wiring graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// Source location in the codebase.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Location {
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
    pub end_line: Option<u32>,
    pub end_column: Option<u32>,
}

impl Location {
    pub fn new(file: PathBuf, line: u32, column: u32) -> Self {
        Self {
            file,
            line,
            column,
            end_line: None,
            end_column: None,
        }
    }

    pub fn with_end(mut self, end_line: u32, end_column: u32) -> Self {
        self.end_line = Some(end_line);
        self.end_column = Some(end_column);
        self
    }
}

/// A node in the wiring graph representing a code element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    /// A function or method definition.
    Function {
        name: String,
        loc: Location,
        is_async: bool,
        is_public: bool,
        params: Vec<String>,
    },

    /// A type/struct/class definition.
    Type {
        name: String,
        loc: Location,
        kind: TypeKind,
    },

    /// An event handler (onClick, on_message, etc.).
    Handler {
        name: String,
        event: String,
        loc: Location,
    },

    /// An HTTP route definition.
    Route {
        path: String,
        method: HttpMethod,
        handler: Option<NodeId>,
        loc: Location,
    },

    /// An entry point into the application.
    EntryPoint {
        kind: EntryKind,
        loc: Location,
    },

    /// An observable effect (I/O, network, etc.).
    Effect {
        kind: EffectKind,
        loc: Location,
    },

    /// A variable or state that can hold data.
    Variable {
        name: String,
        loc: Location,
    },

    /// A module or file.
    Module {
        name: String,
        path: PathBuf,
    },
}

impl Node {
    pub fn location(&self) -> Option<&Location> {
        match self {
            Node::Function { loc, .. }
            | Node::Type { loc, .. }
            | Node::Handler { loc, .. }
            | Node::Route { loc, .. }
            | Node::EntryPoint { loc, .. }
            | Node::Effect { loc, .. }
            | Node::Variable { loc, .. } => Some(loc),
            Node::Module { .. } => None,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Node::Function { name, .. }
            | Node::Type { name, .. }
            | Node::Handler { name, .. }
            | Node::Variable { name, .. }
            | Node::Module { name, .. } => name,
            Node::Route { path, .. } => path,
            Node::EntryPoint { kind, .. } => kind.as_str(),
            Node::Effect { kind, .. } => kind.as_str(),
        }
    }

    pub fn is_entry_point(&self) -> bool {
        matches!(self, Node::EntryPoint { .. })
    }

    pub fn is_effect(&self) -> bool {
        matches!(self, Node::Effect { .. })
    }
}

/// Kind of type definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeKind {
    Struct,
    Enum,
    Trait,
    Class,
    Interface,
    TypeAlias,
}

/// HTTP method for route definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
    Any,
}

/// Kind of entry point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryKind {
    /// fn main() or equivalent
    Main,
    /// #[tokio::main], @main, etc.
    AsyncMain,
    /// #[test], def test_*, etc.
    Test { name: String },
    /// Exported/public function
    ExportedFunction { name: String },
    /// HTTP request handler
    HttpHandler { method: HttpMethod, path: String },
    /// Event listener
    EventHandler { event: String },
    /// Scheduled task (cron, etc.)
    ScheduledTask { schedule: String },
    /// Signal handler (SIGTERM, etc.)
    SignalHandler { signal: String },
    /// Library entry point
    LibraryExport { name: String },
}

impl EntryKind {
    pub fn as_str(&self) -> &str {
        match self {
            EntryKind::Main => "main",
            EntryKind::AsyncMain => "async_main",
            EntryKind::Test { name } => name,
            EntryKind::ExportedFunction { name } => name,
            EntryKind::HttpHandler { .. } => "http_handler",
            EntryKind::EventHandler { .. } => "event_handler",
            EntryKind::ScheduledTask { .. } => "scheduled_task",
            EntryKind::SignalHandler { .. } => "signal_handler",
            EntryKind::LibraryExport { name } => name,
        }
    }
}

/// Kind of observable effect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectKind {
    /// Writing to stdout (println!, console.log, print())
    Stdout,
    /// Writing to stderr
    Stderr,
    /// Writing to a file
    FileWrite { path: Option<String> },
    /// Network request (HTTP, gRPC, WebSocket send)
    NetworkRequest { kind: NetworkKind },
    /// Database write operation
    DatabaseWrite { operation: String },
    /// Spawning a process
    ProcessSpawn { command: Option<String> },
    /// System call
    SystemCall { call: String },
    /// UI rendering (DOM update, widget draw)
    UiRender { component: Option<String> },
    /// Return from an entry point
    Return,
    /// Exit/abort
    Exit { code: Option<i32> },
    /// Panic/throw
    Panic,
}

impl EffectKind {
    pub fn as_str(&self) -> &str {
        match self {
            EffectKind::Stdout => "stdout",
            EffectKind::Stderr => "stderr",
            EffectKind::FileWrite { .. } => "file_write",
            EffectKind::NetworkRequest { .. } => "network_request",
            EffectKind::DatabaseWrite { .. } => "database_write",
            EffectKind::ProcessSpawn { .. } => "process_spawn",
            EffectKind::SystemCall { .. } => "system_call",
            EffectKind::UiRender { .. } => "ui_render",
            EffectKind::Return => "return",
            EffectKind::Exit { .. } => "exit",
            EffectKind::Panic => "panic",
        }
    }
}

/// Kind of network operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkKind {
    Http,
    Grpc,
    WebSocket,
    Tcp,
    Udp,
    Other(String),
}

/// An edge in the wiring graph representing a relationship between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Edge {
    /// A function call from one node to another.
    Calls {
        from: NodeId,
        to: NodeId,
        loc: Location,
    },

    /// A reference (but not call) from one node to another.
    References {
        from: NodeId,
        to: NodeId,
        loc: Location,
    },

    /// Data flows from one node to another.
    DataFlow {
        from: NodeId,
        to: NodeId,
        via: String,
        loc: Location,
    },

    /// A handler is registered with a framework component.
    Registers {
        handler: NodeId,
        with: NodeId,
        framework: String,
        loc: Location,
    },

    /// An async await relationship.
    Awaits {
        from: NodeId,
        to: NodeId,
        loc: Location,
    },

    /// Spawning a task or thread.
    Spawns {
        from: NodeId,
        to: NodeId,
        loc: Location,
    },

    /// Imports or extends relationship.
    Imports {
        from: NodeId,
        to: NodeId,
    },

    /// Contains relationship (module contains function).
    Contains {
        parent: NodeId,
        child: NodeId,
    },
}

impl Edge {
    pub fn source(&self) -> NodeId {
        match self {
            Edge::Calls { from, .. }
            | Edge::References { from, .. }
            | Edge::DataFlow { from, .. }
            | Edge::Awaits { from, .. }
            | Edge::Spawns { from, .. }
            | Edge::Imports { from, .. } => *from,
            Edge::Registers { handler, .. } => *handler,
            Edge::Contains { parent, .. } => *parent,
        }
    }

    pub fn target(&self) -> NodeId {
        match self {
            Edge::Calls { to, .. }
            | Edge::References { to, .. }
            | Edge::DataFlow { to, .. }
            | Edge::Awaits { to, .. }
            | Edge::Spawns { to, .. }
            | Edge::Imports { to, .. } => *to,
            Edge::Registers { with, .. } => *with,
            Edge::Contains { child, .. } => *child,
        }
    }

    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            Edge::Calls { .. } | Edge::Awaits { .. } | Edge::Spawns { .. }
        )
    }
}

/// The wiring graph representing an application's structure.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct WiringGraph {
    nodes: IndexMap<NodeId, Node>,
    edges: Vec<Edge>,
    next_id: u32,

    // Cached indices for fast lookup
    #[serde(skip)]
    outgoing: FxHashMap<NodeId, Vec<usize>>,
    #[serde(skip)]
    incoming: FxHashMap<NodeId, Vec<usize>>,
    #[serde(skip)]
    by_name: FxHashMap<String, Vec<NodeId>>,
}

impl WiringGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph and return its ID.
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        // Index by name
        let name = node.name().to_string();
        self.by_name.entry(name).or_default().push(id);

        self.nodes.insert(id, node);
        id
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: Edge) {
        let idx = self.edges.len();
        let source = edge.source();
        let target = edge.target();

        self.outgoing.entry(source).or_default().push(idx);
        self.incoming.entry(target).or_default().push(idx);

        self.edges.push(edge);
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Iterate over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().map(|(id, node)| (*id, node))
    }

    /// Iterate over all edges.
    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.iter()
    }

    /// Get all entry points.
    pub fn entry_points(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.is_entry_point())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all effect nodes.
    pub fn effects(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.is_effect())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all function nodes.
    pub fn functions(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, node)| matches!(node, Node::Function { .. }))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get outgoing edges from a node.
    pub fn outgoing_edges(&self, id: NodeId) -> impl Iterator<Item = &Edge> {
        self.outgoing
            .get(&id)
            .into_iter()
            .flatten()
            .filter_map(|&idx| self.edges.get(idx))
    }

    /// Get incoming edges to a node.
    pub fn incoming_edges(&self, id: NodeId) -> impl Iterator<Item = &Edge> {
        self.incoming
            .get(&id)
            .into_iter()
            .flatten()
            .filter_map(|&idx| self.edges.get(idx))
    }

    /// Get nodes that this node calls (direct successors via control flow).
    pub fn callees(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.outgoing_edges(id)
            .filter(|e| e.is_control_flow())
            .map(|e| e.target())
    }

    /// Get nodes that call this node (direct predecessors via control flow).
    pub fn callers(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.incoming_edges(id)
            .filter(|e| e.is_control_flow())
            .map(|e| e.source())
    }

    /// Find nodes by name.
    pub fn find_by_name(&self, name: &str) -> impl Iterator<Item = NodeId> + '_ {
        self.by_name.get(name).into_iter().flatten().copied()
    }

    /// Check if there is a path from source to target.
    pub fn path_exists(&self, source: NodeId, target: NodeId) -> bool {
        if source == target {
            return true;
        }

        let mut visited = FxHashSet::default();
        let mut stack = vec![source];

        while let Some(current) = stack.pop() {
            if current == target {
                return true;
            }

            if visited.insert(current) {
                for callee in self.callees(current) {
                    if !visited.contains(&callee) {
                        stack.push(callee);
                    }
                }
            }
        }

        false
    }

    /// Get all nodes reachable from the given source.
    pub fn reachable_from(&self, source: NodeId) -> FxHashSet<NodeId> {
        let mut visited = FxHashSet::default();
        let mut stack = vec![source];

        while let Some(current) = stack.pop() {
            if visited.insert(current) {
                for callee in self.callees(current) {
                    if !visited.contains(&callee) {
                        stack.push(callee);
                    }
                }
            }
        }

        visited
    }

    /// Get all nodes reachable from any entry point.
    pub fn reachable_from_entries(&self) -> FxHashSet<NodeId> {
        let mut reachable = FxHashSet::default();
        for entry in self.entry_points() {
            reachable.extend(self.reachable_from(entry));
        }
        reachable
    }

    /// Rebuild the internal indices (call after deserialization).
    pub fn rebuild_indices(&mut self) {
        self.outgoing.clear();
        self.incoming.clear();
        self.by_name.clear();

        for (id, node) in &self.nodes {
            let name = node.name().to_string();
            self.by_name.entry(name).or_default().push(*id);
        }

        for (idx, edge) in self.edges.iter().enumerate() {
            self.outgoing.entry(edge.source()).or_default().push(idx);
            self.incoming.entry(edge.target()).or_default().push(idx);
        }
    }

    /// Get statistics about the graph.
    pub fn stats(&self) -> GraphStats {
        let mut stats = GraphStats::default();
        stats.total_nodes = self.nodes.len();
        stats.total_edges = self.edges.len();

        for (_, node) in &self.nodes {
            match node {
                Node::Function { .. } => stats.functions += 1,
                Node::Handler { .. } => stats.handlers += 1,
                Node::Route { .. } => stats.routes += 1,
                Node::EntryPoint { .. } => stats.entry_points += 1,
                Node::Effect { .. } => stats.effects += 1,
                Node::Type { .. } => stats.types += 1,
                Node::Variable { .. } => stats.variables += 1,
                Node::Module { .. } => stats.modules += 1,
            }
        }

        for edge in &self.edges {
            match edge {
                Edge::Calls { .. } => stats.call_edges += 1,
                Edge::DataFlow { .. } => stats.data_flow_edges += 1,
                Edge::Registers { .. } => stats.registration_edges += 1,
                _ => stats.other_edges += 1,
            }
        }

        stats
    }
}

/// Statistics about the wiring graph.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub functions: usize,
    pub handlers: usize,
    pub routes: usize,
    pub entry_points: usize,
    pub effects: usize,
    pub types: usize,
    pub variables: usize,
    pub modules: usize,
    pub call_edges: usize,
    pub data_flow_edges: usize,
    pub registration_edges: usize,
    pub other_edges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node_and_edge() {
        let mut graph = WiringGraph::new();

        let main = graph.add_node(Node::EntryPoint {
            kind: EntryKind::Main,
            loc: Location::new("src/main.rs".into(), 1, 1),
        });

        let foo = graph.add_node(Node::Function {
            name: "foo".to_string(),
            loc: Location::new("src/main.rs".into(), 5, 1),
            is_async: false,
            is_public: false,
            params: vec![],
        });

        graph.add_edge(Edge::Calls {
            from: main,
            to: foo,
            loc: Location::new("src/main.rs".into(), 2, 5),
        });

        assert_eq!(graph.nodes().count(), 2);
        assert_eq!(graph.edges().count(), 1);
        assert!(graph.path_exists(main, foo));
        assert!(!graph.path_exists(foo, main));
    }

    #[test]
    fn test_reachability() {
        let mut graph = WiringGraph::new();

        let main = graph.add_node(Node::EntryPoint {
            kind: EntryKind::Main,
            loc: Location::new("src/main.rs".into(), 1, 1),
        });

        let a = graph.add_node(Node::Function {
            name: "a".to_string(),
            loc: Location::new("src/main.rs".into(), 5, 1),
            is_async: false,
            is_public: false,
            params: vec![],
        });

        let b = graph.add_node(Node::Function {
            name: "b".to_string(),
            loc: Location::new("src/main.rs".into(), 10, 1),
            is_async: false,
            is_public: false,
            params: vec![],
        });

        let orphan = graph.add_node(Node::Function {
            name: "orphan".to_string(),
            loc: Location::new("src/main.rs".into(), 15, 1),
            is_async: false,
            is_public: false,
            params: vec![],
        });

        // main -> a -> b
        graph.add_edge(Edge::Calls {
            from: main,
            to: a,
            loc: Location::new("src/main.rs".into(), 2, 5),
        });
        graph.add_edge(Edge::Calls {
            from: a,
            to: b,
            loc: Location::new("src/main.rs".into(), 6, 5),
        });

        let reachable = graph.reachable_from_entries();
        assert!(reachable.contains(&main));
        assert!(reachable.contains(&a));
        assert!(reachable.contains(&b));
        assert!(!reachable.contains(&orphan));
    }
}
