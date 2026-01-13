//! SMT-LIB commands
//!
//! Represents and parses SMT-LIB 2.6 commands.

use crate::sexp::{ParseError, SExpr};

/// An SMT-LIB sort
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sort {
    /// A simple sort (Bool, Int, Real, etc.)
    Simple(String),
    /// A parameterized sort (Array Int Int, BitVec 32, etc.)
    Parameterized(String, Vec<Sort>),
    /// An indexed sort (_ BitVec 32)
    Indexed(String, Vec<String>),
}

impl Sort {
    /// Parse a sort from an S-expression
    pub fn from_sexp(sexp: &SExpr) -> Result<Self, ParseError> {
        match sexp {
            SExpr::Symbol(name) => Ok(Sort::Simple(name.clone())),
            SExpr::List(items) if !items.is_empty() => {
                // Check for indexed identifier (_ name index+)
                if items[0].is_symbol("_") && items.len() >= 2 {
                    let name = items[1]
                        .as_symbol()
                        .ok_or_else(|| ParseError::new("Expected symbol in indexed sort"))?;
                    let indices: Result<Vec<_>, _> = items[2..]
                        .iter()
                        .map(|s| match s {
                            SExpr::Numeral(n) => Ok(n.clone()),
                            SExpr::Symbol(s) => Ok(s.clone()),
                            _ => Err(ParseError::new(
                                "Expected numeral or symbol in indexed sort",
                            )),
                        })
                        .collect();
                    Ok(Sort::Indexed(name.to_string(), indices?))
                } else {
                    // Parameterized sort
                    let name = items[0]
                        .as_symbol()
                        .ok_or_else(|| ParseError::new("Expected symbol as sort constructor"))?;
                    let params: Result<Vec<_>, _> =
                        items[1..].iter().map(Sort::from_sexp).collect();
                    Ok(Sort::Parameterized(name.to_string(), params?))
                }
            }
            _ => Err(ParseError::new(format!("Invalid sort: {sexp}"))),
        }
    }
}

/// An SMT-LIB term
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// A constant: true, false
    Const(Constant),
    /// A variable or uninterpreted constant
    Symbol(String),
    /// Function application: (f arg1 arg2 ...)
    App(String, Vec<Term>),
    /// Let binding: (let ((x e1) (y e2)) body)
    Let(Vec<(String, Term)>, Box<Term>),
    /// Quantifier: (forall ((x Int)) body)
    Forall(Vec<(String, Sort)>, Box<Term>),
    /// Quantifier: (exists ((x Int)) body)
    Exists(Vec<(String, Sort)>, Box<Term>),
    /// Annotated term: (! term :named foo)
    Annotated(Box<Term>, Vec<(String, SExpr)>),
}

/// Constant values
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    /// Boolean true
    True,
    /// Boolean false
    False,
    /// Numeral
    Numeral(String),
    /// Decimal
    Decimal(String),
    /// Hexadecimal bitvector
    Hexadecimal(String),
    /// Binary bitvector
    Binary(String),
    /// String literal
    String(String),
}

impl Term {
    /// Parse a term from an S-expression
    pub fn from_sexp(sexp: &SExpr) -> Result<Self, ParseError> {
        match sexp {
            SExpr::True => Ok(Term::Const(Constant::True)),
            SExpr::False => Ok(Term::Const(Constant::False)),
            SExpr::Numeral(n) => Ok(Term::Const(Constant::Numeral(n.clone()))),
            SExpr::Decimal(d) => Ok(Term::Const(Constant::Decimal(d.clone()))),
            SExpr::Hexadecimal(h) => Ok(Term::Const(Constant::Hexadecimal(h.clone()))),
            SExpr::Binary(b) => Ok(Term::Const(Constant::Binary(b.clone()))),
            SExpr::String(s) => Ok(Term::Const(Constant::String(s.clone()))),
            SExpr::Symbol(s) => Ok(Term::Symbol(s.clone())),
            SExpr::Keyword(_) => Err(ParseError::new("Unexpected keyword in term")),
            SExpr::List(items) if items.is_empty() => {
                Err(ParseError::new("Empty list is not a valid term"))
            }
            SExpr::List(items) => {
                // Check for special forms
                if let Some(head) = items[0].as_symbol() {
                    match head {
                        "let" => Self::parse_let(items),
                        "forall" => Self::parse_quantifier(items, true),
                        "exists" => Self::parse_quantifier(items, false),
                        "!" => Self::parse_annotated(items),
                        "_" => Self::parse_indexed_identifier(items),
                        _ => Self::parse_application(items),
                    }
                } else if let SExpr::List(_) = &items[0] {
                    // ((as const ...) args) pattern or nested application
                    Self::parse_application(items)
                } else {
                    Err(ParseError::new(format!("Invalid term head: {}", items[0])))
                }
            }
        }
    }

    fn parse_let(items: &[SExpr]) -> Result<Self, ParseError> {
        if items.len() != 3 {
            return Err(ParseError::new("let requires bindings and body"));
        }
        let bindings_sexp = items[1]
            .as_list()
            .ok_or_else(|| ParseError::new("let bindings must be a list"))?;

        let mut bindings = Vec::new();
        for binding in bindings_sexp {
            let binding_list = binding
                .as_list()
                .ok_or_else(|| ParseError::new("let binding must be a list"))?;
            if binding_list.len() != 2 {
                return Err(ParseError::new("let binding must have name and value"));
            }
            let name = binding_list[0]
                .as_symbol()
                .ok_or_else(|| ParseError::new("let binding name must be a symbol"))?;
            let value = Term::from_sexp(&binding_list[1])?;
            bindings.push((name.to_string(), value));
        }

        let body = Term::from_sexp(&items[2])?;
        Ok(Term::Let(bindings, Box::new(body)))
    }

    fn parse_quantifier(items: &[SExpr], is_forall: bool) -> Result<Self, ParseError> {
        if items.len() != 3 {
            return Err(ParseError::new("quantifier requires bindings and body"));
        }
        let bindings_sexp = items[1]
            .as_list()
            .ok_or_else(|| ParseError::new("quantifier bindings must be a list"))?;

        let mut bindings = Vec::new();
        for binding in bindings_sexp {
            let binding_list = binding
                .as_list()
                .ok_or_else(|| ParseError::new("quantifier binding must be a list"))?;
            if binding_list.len() != 2 {
                return Err(ParseError::new(
                    "quantifier binding must have name and sort",
                ));
            }
            let name = binding_list[0]
                .as_symbol()
                .ok_or_else(|| ParseError::new("quantifier binding name must be a symbol"))?;
            let sort = Sort::from_sexp(&binding_list[1])?;
            bindings.push((name.to_string(), sort));
        }

        let body = Term::from_sexp(&items[2])?;
        if is_forall {
            Ok(Term::Forall(bindings, Box::new(body)))
        } else {
            Ok(Term::Exists(bindings, Box::new(body)))
        }
    }

    fn parse_annotated(items: &[SExpr]) -> Result<Self, ParseError> {
        if items.len() < 2 {
            return Err(ParseError::new("annotation requires term"));
        }
        let term = Term::from_sexp(&items[1])?;

        let mut annotations = Vec::new();
        let mut i = 2;
        while i < items.len() {
            if let SExpr::Keyword(k) = &items[i] {
                if i + 1 < items.len() {
                    annotations.push((k.clone(), items[i + 1].clone()));
                    i += 2;
                } else {
                    return Err(ParseError::new("annotation keyword requires value"));
                }
            } else {
                return Err(ParseError::new("expected keyword in annotation"));
            }
        }

        Ok(Term::Annotated(Box::new(term), annotations))
    }

    fn parse_indexed_identifier(items: &[SExpr]) -> Result<Self, ParseError> {
        // (_ symbol index+) - indexed identifier as a term
        if items.len() < 2 {
            return Err(ParseError::new("indexed identifier requires name"));
        }
        let name = items[1]
            .as_symbol()
            .ok_or_else(|| ParseError::new("indexed identifier name must be symbol"))?;

        let indices: Vec<String> = items[2..]
            .iter()
            .filter_map(|s| match s {
                SExpr::Numeral(n) => Some(n.clone()),
                SExpr::Symbol(s) => Some(s.clone()),
                _ => None,
            })
            .collect();

        // Return as a symbol with the full indexed name
        let full_name = format!("(_ {} {})", name, indices.join(" "));
        Ok(Term::Symbol(full_name))
    }

    fn parse_application(items: &[SExpr]) -> Result<Self, ParseError> {
        if items.is_empty() {
            return Err(ParseError::new("empty application"));
        }

        // Handle indexed function names like (_ extract 7 0)
        let (func_name, args_start) = if let SExpr::List(head_items) = &items[0] {
            if !head_items.is_empty() && head_items[0].is_symbol("_") {
                // Indexed function
                let name = head_items
                    .get(1)
                    .and_then(|s| s.as_symbol())
                    .ok_or_else(|| ParseError::new("indexed identifier name must be symbol"))?;
                let indices: Vec<String> = head_items[2..]
                    .iter()
                    .filter_map(|s| match s {
                        SExpr::Numeral(n) => Some(n.clone()),
                        SExpr::Symbol(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
                (format!("(_ {} {})", name, indices.join(" ")), 1)
            } else if !head_items.is_empty() && head_items[0].is_symbol("as") {
                // (as const (Array Int Int)) pattern
                let name = format!("{}", items[0]);
                (name, 1)
            } else {
                return Err(ParseError::new("invalid function in application"));
            }
        } else {
            let name = items[0]
                .as_symbol()
                .ok_or_else(|| ParseError::new("function name must be symbol"))?
                .to_string();
            (name, 1)
        };

        let args: Result<Vec<_>, _> = items[args_start..].iter().map(Term::from_sexp).collect();

        Ok(Term::App(func_name, args?))
    }
}

/// A function declaration: (name, parameters, return sort)
/// Used in define-funs-rec for mutually recursive function definitions.
pub type FuncDeclaration = (String, Vec<(String, Sort)>, Sort);

/// A selector declaration in a datatype constructor: (name, sort)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectorDec {
    /// The selector name (accessor function)
    pub name: String,
    /// The sort of the field
    pub sort: Sort,
}

/// A constructor declaration in a datatype: (name, selectors)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructorDec {
    /// The constructor name
    pub name: String,
    /// The selectors (fields) of this constructor
    pub selectors: Vec<SelectorDec>,
}

/// A datatype declaration: list of constructors
/// Note: Parametric datatypes (par ...) are not yet supported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatatypeDec {
    /// The constructors for this datatype
    pub constructors: Vec<ConstructorDec>,
}

/// A sort declaration for declare-datatypes: (name, arity)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortDec {
    /// The sort name
    pub name: String,
    /// The arity (0 for non-parametric sorts)
    pub arity: u32,
}

/// An SMT-LIB command
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// `(set-logic <symbol>)`
    SetLogic(String),
    /// `(set-option <keyword> <value>)`
    SetOption(String, SExpr),
    /// `(set-info <keyword> <value>)`
    SetInfo(String, SExpr),
    /// `(declare-sort <symbol> <numeral>)`
    DeclareSort(String, u32),
    /// `(define-sort <symbol> (<symbol>*) <sort>)`
    DefineSort(String, Vec<String>, Sort),
    /// `(declare-datatype <symbol> <datatype_dec>)`
    DeclareDatatype(String, DatatypeDec),
    /// `(declare-datatypes (<sort_dec>+) (<datatype_dec>+))`
    DeclareDatatypes(Vec<SortDec>, Vec<DatatypeDec>),
    /// `(declare-fun <symbol> (<sort>*) <sort>)`
    DeclareFun(String, Vec<Sort>, Sort),
    /// `(declare-const <symbol> <sort>)`
    DeclareConst(String, Sort),
    /// `(define-fun <symbol> (<sorted_var>*) <sort> <term>)`
    DefineFun(String, Vec<(String, Sort)>, Sort, Term),
    /// `(define-fun-rec <symbol> (<sorted_var>*) <sort> <term>)`
    DefineFunRec(String, Vec<(String, Sort)>, Sort, Term),
    /// `(define-funs-rec (<func_dec>+) (<term>+))`
    /// where `func_dec = (<symbol> (<sorted_var>*) <sort>)`
    DefineFunsRec(Vec<FuncDeclaration>, Vec<Term>),
    /// `(assert <term>)`
    Assert(Term),
    /// `(check-sat)`
    CheckSat,
    /// `(check-sat-assuming (<literal>*))`
    CheckSatAssuming(Vec<Term>),
    /// `(get-model)`
    GetModel,
    /// `(get-value (<term>+))`
    GetValue(Vec<Term>),
    /// `(get-unsat-core)`
    GetUnsatCore,
    /// `(get-unsat-assumptions)`
    GetUnsatAssumptions,
    /// `(get-proof)`
    GetProof,
    /// `(get-assertions)`
    GetAssertions,
    /// `(get-assignment)`
    GetAssignment,
    /// `(get-info <keyword>)`
    GetInfo(String),
    /// `(get-option <keyword>)`
    GetOption(String),
    /// `(push <numeral>)`
    Push(u32),
    /// `(pop <numeral>)`
    Pop(u32),
    /// `(reset)`
    Reset,
    /// `(reset-assertions)`
    ResetAssertions,
    /// `(exit)`
    Exit,
    /// `(echo <string>)`
    Echo(String),
    /// `(simplify <term>)` - Z3 extension
    Simplify(Term),
}

impl SelectorDec {
    /// Parse a selector declaration: (name sort)
    fn from_sexp(sexp: &SExpr) -> Result<Self, ParseError> {
        let items = sexp
            .as_list()
            .ok_or_else(|| ParseError::new("selector must be a list"))?;
        if items.len() != 2 {
            return Err(ParseError::new("selector must be (name sort)"));
        }
        let name = items[0]
            .as_symbol()
            .ok_or_else(|| ParseError::new("selector name must be symbol"))?;
        let sort = Sort::from_sexp(&items[1])?;
        Ok(SelectorDec {
            name: name.to_string(),
            sort,
        })
    }
}

impl ConstructorDec {
    /// Parse a constructor declaration: (name selector*)
    fn from_sexp(sexp: &SExpr) -> Result<Self, ParseError> {
        let items = sexp
            .as_list()
            .ok_or_else(|| ParseError::new("constructor must be a list"))?;
        if items.is_empty() {
            return Err(ParseError::new("constructor requires name"));
        }
        let name = items[0]
            .as_symbol()
            .ok_or_else(|| ParseError::new("constructor name must be symbol"))?;
        let selectors: Result<Vec<_>, _> = items[1..].iter().map(SelectorDec::from_sexp).collect();
        Ok(ConstructorDec {
            name: name.to_string(),
            selectors: selectors?,
        })
    }
}

impl DatatypeDec {
    /// Parse a datatype declaration: (constructor+) or (par (...) (constructor+))
    fn from_sexp(sexp: &SExpr) -> Result<Self, ParseError> {
        let items = sexp
            .as_list()
            .ok_or_else(|| ParseError::new("datatype declaration must be a list"))?;
        if items.is_empty() {
            return Err(ParseError::new(
                "datatype requires at least one constructor",
            ));
        }

        // Check for parametric datatype
        if items[0].is_symbol("par") {
            return Err(ParseError::new(
                "parametric datatypes (par ...) are not yet supported",
            ));
        }

        // Non-parametric: list of constructors
        let constructors: Result<Vec<_>, _> = items.iter().map(ConstructorDec::from_sexp).collect();
        Ok(DatatypeDec {
            constructors: constructors?,
        })
    }
}

impl SortDec {
    /// Parse a sort declaration: (name arity)
    fn from_sexp(sexp: &SExpr) -> Result<Self, ParseError> {
        let items = sexp
            .as_list()
            .ok_or_else(|| ParseError::new("sort declaration must be a list"))?;
        if items.len() != 2 {
            return Err(ParseError::new("sort declaration must be (name arity)"));
        }
        let name = items[0]
            .as_symbol()
            .ok_or_else(|| ParseError::new("sort name must be symbol"))?;
        let arity = items[1]
            .as_numeral()
            .and_then(|n| n.parse::<u32>().ok())
            .ok_or_else(|| ParseError::new("sort arity must be numeral"))?;
        Ok(SortDec {
            name: name.to_string(),
            arity,
        })
    }
}

impl Command {
    /// Parse a command from an S-expression
    pub fn from_sexp(sexp: &SExpr) -> Result<Self, ParseError> {
        let items = sexp
            .as_list()
            .ok_or_else(|| ParseError::new("Command must be a list"))?;

        if items.is_empty() {
            return Err(ParseError::new("Empty command"));
        }

        let cmd = items[0]
            .as_symbol()
            .ok_or_else(|| ParseError::new("Command name must be a symbol"))?;

        match cmd {
            "set-logic" => {
                let logic = items
                    .get(1)
                    .and_then(|s| s.as_symbol())
                    .ok_or_else(|| ParseError::new("set-logic requires logic name"))?;
                Ok(Command::SetLogic(logic.to_string()))
            }
            "set-option" => {
                if items.len() < 3 {
                    return Err(ParseError::new("set-option requires keyword and value"));
                }
                let keyword = match &items[1] {
                    SExpr::Keyword(k) => k.clone(),
                    _ => return Err(ParseError::new("set-option requires keyword")),
                };
                Ok(Command::SetOption(keyword, items[2].clone()))
            }
            "set-info" => {
                if items.len() < 3 {
                    return Err(ParseError::new("set-info requires keyword and value"));
                }
                let keyword = match &items[1] {
                    SExpr::Keyword(k) => k.clone(),
                    _ => return Err(ParseError::new("set-info requires keyword")),
                };
                Ok(Command::SetInfo(keyword, items[2].clone()))
            }
            "declare-sort" => {
                let name = items
                    .get(1)
                    .and_then(|s| s.as_symbol())
                    .ok_or_else(|| ParseError::new("declare-sort requires name"))?;
                let arity = items
                    .get(2)
                    .and_then(|s| s.as_numeral())
                    .and_then(|n| n.parse::<u32>().ok())
                    .unwrap_or(0);
                Ok(Command::DeclareSort(name.to_string(), arity))
            }
            "define-sort" => {
                let name = items
                    .get(1)
                    .and_then(|s| s.as_symbol())
                    .ok_or_else(|| ParseError::new("define-sort requires name"))?;
                let params = items
                    .get(2)
                    .and_then(|s| s.as_list())
                    .ok_or_else(|| ParseError::new("define-sort requires parameter list"))?;
                let param_names: Result<Vec<_>, _> = params
                    .iter()
                    .map(|p| {
                        p.as_symbol()
                            .map(String::from)
                            .ok_or_else(|| ParseError::new("sort parameter must be symbol"))
                    })
                    .collect();
                let sort = items
                    .get(3)
                    .ok_or_else(|| ParseError::new("define-sort requires sort definition"))?;
                Ok(Command::DefineSort(
                    name.to_string(),
                    param_names?,
                    Sort::from_sexp(sort)?,
                ))
            }
            "declare-datatype" => {
                // (declare-datatype name datatype_dec)
                let name = items
                    .get(1)
                    .and_then(|s| s.as_symbol())
                    .ok_or_else(|| ParseError::new("declare-datatype requires name"))?;
                let datatype_dec = items.get(2).ok_or_else(|| {
                    ParseError::new("declare-datatype requires datatype declaration")
                })?;
                Ok(Command::DeclareDatatype(
                    name.to_string(),
                    DatatypeDec::from_sexp(datatype_dec)?,
                ))
            }
            "declare-datatypes" => {
                // (declare-datatypes ((name1 arity1) ...) (datatype_dec1 ...))
                let sort_decs = items.get(1).and_then(|s| s.as_list()).ok_or_else(|| {
                    ParseError::new("declare-datatypes requires sort declarations")
                })?;
                let datatype_decs = items.get(2).and_then(|s| s.as_list()).ok_or_else(|| {
                    ParseError::new("declare-datatypes requires datatype declarations")
                })?;

                if sort_decs.len() != datatype_decs.len() {
                    return Err(ParseError::new(
                        "declare-datatypes: number of sort declarations must match datatype declarations",
                    ));
                }

                let sorts: Result<Vec<_>, _> = sort_decs.iter().map(SortDec::from_sexp).collect();
                let datatypes: Result<Vec<_>, _> =
                    datatype_decs.iter().map(DatatypeDec::from_sexp).collect();

                Ok(Command::DeclareDatatypes(sorts?, datatypes?))
            }
            "declare-fun" => {
                let name = items
                    .get(1)
                    .and_then(|s| s.as_symbol())
                    .ok_or_else(|| ParseError::new("declare-fun requires name"))?;
                let args = items
                    .get(2)
                    .and_then(|s| s.as_list())
                    .ok_or_else(|| ParseError::new("declare-fun requires argument sorts"))?;
                let arg_sorts: Result<Vec<_>, _> = args.iter().map(Sort::from_sexp).collect();
                let ret = items
                    .get(3)
                    .ok_or_else(|| ParseError::new("declare-fun requires return sort"))?;
                Ok(Command::DeclareFun(
                    name.to_string(),
                    arg_sorts?,
                    Sort::from_sexp(ret)?,
                ))
            }
            "declare-const" => {
                let name = items
                    .get(1)
                    .and_then(|s| s.as_symbol())
                    .ok_or_else(|| ParseError::new("declare-const requires name"))?;
                let sort = items
                    .get(2)
                    .ok_or_else(|| ParseError::new("declare-const requires sort"))?;
                Ok(Command::DeclareConst(
                    name.to_string(),
                    Sort::from_sexp(sort)?,
                ))
            }
            "define-fun" => {
                let name = items
                    .get(1)
                    .and_then(|s| s.as_symbol())
                    .ok_or_else(|| ParseError::new("define-fun requires name"))?;
                let params = items
                    .get(2)
                    .and_then(|s| s.as_list())
                    .ok_or_else(|| ParseError::new("define-fun requires parameter list"))?;
                let mut sorted_vars = Vec::new();
                for param in params {
                    let param_list = param
                        .as_list()
                        .ok_or_else(|| ParseError::new("parameter must be (name sort)"))?;
                    if param_list.len() != 2 {
                        return Err(ParseError::new("parameter must be (name sort)"));
                    }
                    let var_name = param_list[0]
                        .as_symbol()
                        .ok_or_else(|| ParseError::new("parameter name must be symbol"))?;
                    let var_sort = Sort::from_sexp(&param_list[1])?;
                    sorted_vars.push((var_name.to_string(), var_sort));
                }
                let ret_sort = items
                    .get(3)
                    .ok_or_else(|| ParseError::new("define-fun requires return sort"))?;
                let body = items
                    .get(4)
                    .ok_or_else(|| ParseError::new("define-fun requires body"))?;
                Ok(Command::DefineFun(
                    name.to_string(),
                    sorted_vars,
                    Sort::from_sexp(ret_sort)?,
                    Term::from_sexp(body)?,
                ))
            }
            "define-fun-rec" => {
                // Same structure as define-fun: (define-fun-rec name ((param sort)*) sort body)
                let name = items
                    .get(1)
                    .and_then(|s| s.as_symbol())
                    .ok_or_else(|| ParseError::new("define-fun-rec requires name"))?;
                let params = items
                    .get(2)
                    .and_then(|s| s.as_list())
                    .ok_or_else(|| ParseError::new("define-fun-rec requires parameter list"))?;
                let mut sorted_vars = Vec::new();
                for param in params {
                    let param_list = param
                        .as_list()
                        .ok_or_else(|| ParseError::new("parameter must be (name sort)"))?;
                    if param_list.len() != 2 {
                        return Err(ParseError::new("parameter must be (name sort)"));
                    }
                    let var_name = param_list[0]
                        .as_symbol()
                        .ok_or_else(|| ParseError::new("parameter name must be symbol"))?;
                    let var_sort = Sort::from_sexp(&param_list[1])?;
                    sorted_vars.push((var_name.to_string(), var_sort));
                }
                let ret_sort = items
                    .get(3)
                    .ok_or_else(|| ParseError::new("define-fun-rec requires return sort"))?;
                let body = items
                    .get(4)
                    .ok_or_else(|| ParseError::new("define-fun-rec requires body"))?;
                Ok(Command::DefineFunRec(
                    name.to_string(),
                    sorted_vars,
                    Sort::from_sexp(ret_sort)?,
                    Term::from_sexp(body)?,
                ))
            }
            "define-funs-rec" => {
                // (define-funs-rec ((f1 ((x T)) T) (f2 ((y T)) T)) (body1 body2))
                let func_decs = items.get(1).and_then(|s| s.as_list()).ok_or_else(|| {
                    ParseError::new("define-funs-rec requires function declarations")
                })?;
                let bodies = items
                    .get(2)
                    .and_then(|s| s.as_list())
                    .ok_or_else(|| ParseError::new("define-funs-rec requires function bodies"))?;

                if func_decs.len() != bodies.len() {
                    return Err(ParseError::new(
                        "define-funs-rec: number of declarations must match number of bodies",
                    ));
                }

                let mut declarations = Vec::new();
                for func_dec in func_decs {
                    let dec_list = func_dec
                        .as_list()
                        .ok_or_else(|| ParseError::new("function declaration must be a list"))?;
                    if dec_list.len() != 3 {
                        return Err(ParseError::new(
                            "function declaration must be (name ((param sort)*) sort)",
                        ));
                    }
                    let name = dec_list[0]
                        .as_symbol()
                        .ok_or_else(|| ParseError::new("function name must be symbol"))?;
                    let params = dec_list[1]
                        .as_list()
                        .ok_or_else(|| ParseError::new("parameters must be a list"))?;
                    let mut sorted_vars = Vec::new();
                    for param in params {
                        let param_list = param
                            .as_list()
                            .ok_or_else(|| ParseError::new("parameter must be (name sort)"))?;
                        if param_list.len() != 2 {
                            return Err(ParseError::new("parameter must be (name sort)"));
                        }
                        let var_name = param_list[0]
                            .as_symbol()
                            .ok_or_else(|| ParseError::new("parameter name must be symbol"))?;
                        let var_sort = Sort::from_sexp(&param_list[1])?;
                        sorted_vars.push((var_name.to_string(), var_sort));
                    }
                    let ret_sort = Sort::from_sexp(&dec_list[2])?;
                    declarations.push((name.to_string(), sorted_vars, ret_sort));
                }

                let parsed_bodies: Result<Vec<_>, _> = bodies.iter().map(Term::from_sexp).collect();
                Ok(Command::DefineFunsRec(declarations, parsed_bodies?))
            }
            "assert" => {
                let term = items
                    .get(1)
                    .ok_or_else(|| ParseError::new("assert requires term"))?;
                Ok(Command::Assert(Term::from_sexp(term)?))
            }
            "check-sat" => Ok(Command::CheckSat),
            "check-sat-assuming" => {
                let lits = items
                    .get(1)
                    .and_then(|s| s.as_list())
                    .ok_or_else(|| ParseError::new("check-sat-assuming requires literal list"))?;
                let terms: Result<Vec<_>, _> = lits.iter().map(Term::from_sexp).collect();
                Ok(Command::CheckSatAssuming(terms?))
            }
            "get-model" => Ok(Command::GetModel),
            "get-value" => {
                let terms = items
                    .get(1)
                    .and_then(|s| s.as_list())
                    .ok_or_else(|| ParseError::new("get-value requires term list"))?;
                let parsed: Result<Vec<_>, _> = terms.iter().map(Term::from_sexp).collect();
                Ok(Command::GetValue(parsed?))
            }
            "get-unsat-core" => Ok(Command::GetUnsatCore),
            "get-unsat-assumptions" => Ok(Command::GetUnsatAssumptions),
            "get-proof" => Ok(Command::GetProof),
            "get-assertions" => Ok(Command::GetAssertions),
            "get-assignment" => Ok(Command::GetAssignment),
            "get-info" => {
                let keyword = match items.get(1) {
                    Some(SExpr::Keyword(k)) => k.clone(),
                    _ => return Err(ParseError::new("get-info requires keyword")),
                };
                Ok(Command::GetInfo(keyword))
            }
            "get-option" => {
                let keyword = match items.get(1) {
                    Some(SExpr::Keyword(k)) => k.clone(),
                    _ => return Err(ParseError::new("get-option requires keyword")),
                };
                Ok(Command::GetOption(keyword))
            }
            "push" => {
                let n = items
                    .get(1)
                    .and_then(|s| s.as_numeral())
                    .and_then(|n| n.parse::<u32>().ok())
                    .unwrap_or(1);
                Ok(Command::Push(n))
            }
            "pop" => {
                let n = items
                    .get(1)
                    .and_then(|s| s.as_numeral())
                    .and_then(|n| n.parse::<u32>().ok())
                    .unwrap_or(1);
                Ok(Command::Pop(n))
            }
            "reset" => Ok(Command::Reset),
            "reset-assertions" => Ok(Command::ResetAssertions),
            "exit" => Ok(Command::Exit),
            "echo" => {
                let msg = match items.get(1) {
                    Some(SExpr::String(s)) => s.clone(),
                    _ => return Err(ParseError::new("echo requires string")),
                };
                Ok(Command::Echo(msg))
            }
            "simplify" => {
                let term = items
                    .get(1)
                    .ok_or_else(|| ParseError::new("simplify requires term"))?;
                Ok(Command::Simplify(Term::from_sexp(term)?))
            }
            _ => Err(ParseError::new(format!("Unknown command: {cmd}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sexp::parse_sexp;

    #[test]
    fn test_parse_set_logic() {
        let sexp = parse_sexp("(set-logic QF_LIA)").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        assert_eq!(cmd, Command::SetLogic("QF_LIA".to_string()));
    }

    #[test]
    fn test_parse_declare_fun() {
        let sexp = parse_sexp("(declare-fun x () Int)").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        assert_eq!(
            cmd,
            Command::DeclareFun("x".to_string(), vec![], Sort::Simple("Int".to_string()))
        );
    }

    #[test]
    fn test_parse_declare_fun_with_args() {
        let sexp = parse_sexp("(declare-fun f (Int Int) Bool)").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        assert_eq!(
            cmd,
            Command::DeclareFun(
                "f".to_string(),
                vec![
                    Sort::Simple("Int".to_string()),
                    Sort::Simple("Int".to_string())
                ],
                Sort::Simple("Bool".to_string())
            )
        );
    }

    #[test]
    fn test_parse_declare_const() {
        let sexp = parse_sexp("(declare-const y Real)").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        assert_eq!(
            cmd,
            Command::DeclareConst("y".to_string(), Sort::Simple("Real".to_string()))
        );
    }

    #[test]
    fn test_parse_assert() {
        let sexp = parse_sexp("(assert (> x 0))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::Assert(Term::App(name, args)) => {
                assert_eq!(name, ">");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Assert command"),
        }
    }

    #[test]
    fn test_parse_check_sat() {
        let sexp = parse_sexp("(check-sat)").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        assert_eq!(cmd, Command::CheckSat);
    }

    #[test]
    fn test_parse_push_pop() {
        let push = parse_sexp("(push 2)").unwrap();
        assert_eq!(Command::from_sexp(&push).unwrap(), Command::Push(2));

        let pop = parse_sexp("(pop 1)").unwrap();
        assert_eq!(Command::from_sexp(&pop).unwrap(), Command::Pop(1));
    }

    #[test]
    fn test_parse_bitvector_sort() {
        let sexp = parse_sexp("(declare-const bv (_ BitVec 32))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::DeclareConst(name, Sort::Indexed(sort_name, indices)) => {
                assert_eq!(name, "bv");
                assert_eq!(sort_name, "BitVec");
                assert_eq!(indices, vec!["32"]);
            }
            _ => panic!("Expected DeclareConst with indexed sort"),
        }
    }

    #[test]
    fn test_parse_array_sort() {
        let sexp = parse_sexp("(declare-const arr (Array Int Int))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::DeclareConst(name, Sort::Parameterized(sort_name, params)) => {
                assert_eq!(name, "arr");
                assert_eq!(sort_name, "Array");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("Expected DeclareConst with parameterized sort"),
        }
    }

    #[test]
    fn test_parse_let_term() {
        let sexp = parse_sexp("(assert (let ((x 1)) (+ x 2)))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::Assert(Term::Let(bindings, _body)) => {
                assert_eq!(bindings.len(), 1);
                assert_eq!(bindings[0].0, "x");
            }
            _ => panic!("Expected Assert with Let term"),
        }
    }

    #[test]
    fn test_parse_forall() {
        let sexp = parse_sexp("(assert (forall ((x Int)) (> x 0)))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::Assert(Term::Forall(bindings, _body)) => {
                assert_eq!(bindings.len(), 1);
                assert_eq!(bindings[0].0, "x");
            }
            _ => panic!("Expected Assert with Forall term"),
        }
    }

    #[test]
    fn test_parse_define_fun_rec() {
        // Factorial function
        let sexp =
            parse_sexp("(define-fun-rec fact ((n Int)) Int (ite (= n 0) 1 (* n (fact (- n 1)))))");
        let cmd = Command::from_sexp(&sexp.unwrap()).unwrap();
        match cmd {
            Command::DefineFunRec(name, params, ret_sort, _body) => {
                assert_eq!(name, "fact");
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].0, "n");
                assert_eq!(ret_sort, Sort::Simple("Int".to_string()));
            }
            _ => panic!("Expected DefineFunRec command"),
        }
    }

    #[test]
    fn test_parse_define_fun_rec_multiple_params() {
        let sexp = parse_sexp(
            "(define-fun-rec gcd ((a Int) (b Int)) Int (ite (= b 0) a (gcd b (mod a b))))",
        );
        let cmd = Command::from_sexp(&sexp.unwrap()).unwrap();
        match cmd {
            Command::DefineFunRec(name, params, ret_sort, _body) => {
                assert_eq!(name, "gcd");
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].0, "a");
                assert_eq!(params[1].0, "b");
                assert_eq!(ret_sort, Sort::Simple("Int".to_string()));
            }
            _ => panic!("Expected DefineFunRec command"),
        }
    }

    #[test]
    fn test_parse_define_funs_rec() {
        // Mutually recursive even/odd functions
        let sexp = parse_sexp(
            "(define-funs-rec ((even ((n Int)) Bool) (odd ((n Int)) Bool)) \
             ((ite (= n 0) true (odd (- n 1))) (ite (= n 0) false (even (- n 1)))))",
        );
        let cmd = Command::from_sexp(&sexp.unwrap()).unwrap();
        match cmd {
            Command::DefineFunsRec(declarations, bodies) => {
                assert_eq!(declarations.len(), 2);
                assert_eq!(bodies.len(), 2);

                // Check first declaration (even)
                assert_eq!(declarations[0].0, "even");
                assert_eq!(declarations[0].1.len(), 1);
                assert_eq!(declarations[0].1[0].0, "n");
                assert_eq!(declarations[0].2, Sort::Simple("Bool".to_string()));

                // Check second declaration (odd)
                assert_eq!(declarations[1].0, "odd");
                assert_eq!(declarations[1].1.len(), 1);
                assert_eq!(declarations[1].1[0].0, "n");
                assert_eq!(declarations[1].2, Sort::Simple("Bool".to_string()));
            }
            _ => panic!("Expected DefineFunsRec command"),
        }
    }

    #[test]
    fn test_parse_define_funs_rec_mismatch_error() {
        // Mismatched number of declarations and bodies
        let sexp = parse_sexp("(define-funs-rec ((f ((x Int)) Int)) (body1 body2))");
        let result = Command::from_sexp(&sexp.unwrap());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("number of declarations must match"));
    }

    #[test]
    fn test_parse_declare_datatype_simple() {
        // Simple enumeration type
        let sexp = parse_sexp("(declare-datatype Color ((Red) (Green) (Blue)))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::DeclareDatatype(name, datatype_dec) => {
                assert_eq!(name, "Color");
                assert_eq!(datatype_dec.constructors.len(), 3);
                assert_eq!(datatype_dec.constructors[0].name, "Red");
                assert_eq!(datatype_dec.constructors[0].selectors.len(), 0);
                assert_eq!(datatype_dec.constructors[1].name, "Green");
                assert_eq!(datatype_dec.constructors[2].name, "Blue");
            }
            _ => panic!("Expected DeclareDatatype command"),
        }
    }

    #[test]
    fn test_parse_declare_datatype_with_selectors() {
        // Record type with selectors
        let sexp = parse_sexp("(declare-datatype Point ((mk-point (x Int) (y Int))))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::DeclareDatatype(name, datatype_dec) => {
                assert_eq!(name, "Point");
                assert_eq!(datatype_dec.constructors.len(), 1);
                assert_eq!(datatype_dec.constructors[0].name, "mk-point");
                assert_eq!(datatype_dec.constructors[0].selectors.len(), 2);
                assert_eq!(datatype_dec.constructors[0].selectors[0].name, "x");
                assert_eq!(
                    datatype_dec.constructors[0].selectors[0].sort,
                    Sort::Simple("Int".to_string())
                );
                assert_eq!(datatype_dec.constructors[0].selectors[1].name, "y");
            }
            _ => panic!("Expected DeclareDatatype command"),
        }
    }

    #[test]
    fn test_parse_declare_datatype_multiple_constructors() {
        // Option-like type
        let sexp = parse_sexp("(declare-datatype Option ((None) (Some (value Int))))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::DeclareDatatype(name, datatype_dec) => {
                assert_eq!(name, "Option");
                assert_eq!(datatype_dec.constructors.len(), 2);
                assert_eq!(datatype_dec.constructors[0].name, "None");
                assert_eq!(datatype_dec.constructors[0].selectors.len(), 0);
                assert_eq!(datatype_dec.constructors[1].name, "Some");
                assert_eq!(datatype_dec.constructors[1].selectors.len(), 1);
                assert_eq!(datatype_dec.constructors[1].selectors[0].name, "value");
            }
            _ => panic!("Expected DeclareDatatype command"),
        }
    }

    #[test]
    fn test_parse_declare_datatypes() {
        // Multiple datatypes (potentially mutually recursive)
        let sexp = parse_sexp(
            "(declare-datatypes ((Tree 0) (Forest 0)) \
             (((leaf (val Int)) (node (children Forest))) ((nil) (cons (head Tree) (tail Forest)))))",
        )
        .unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::DeclareDatatypes(sort_decs, datatype_decs) => {
                assert_eq!(sort_decs.len(), 2);
                assert_eq!(datatype_decs.len(), 2);

                // Check sort declarations
                assert_eq!(sort_decs[0].name, "Tree");
                assert_eq!(sort_decs[0].arity, 0);
                assert_eq!(sort_decs[1].name, "Forest");
                assert_eq!(sort_decs[1].arity, 0);

                // Check Tree constructors
                assert_eq!(datatype_decs[0].constructors.len(), 2);
                assert_eq!(datatype_decs[0].constructors[0].name, "leaf");
                assert_eq!(datatype_decs[0].constructors[1].name, "node");

                // Check Forest constructors
                assert_eq!(datatype_decs[1].constructors.len(), 2);
                assert_eq!(datatype_decs[1].constructors[0].name, "nil");
                assert_eq!(datatype_decs[1].constructors[1].name, "cons");
            }
            _ => panic!("Expected DeclareDatatypes command"),
        }
    }

    #[test]
    fn test_parse_declare_datatypes_mismatch_error() {
        // Mismatched number of sort declarations and datatype declarations
        let sexp = parse_sexp("(declare-datatypes ((T 0)) (((a)) ((b))))");
        let result = Command::from_sexp(&sexp.unwrap());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("number of sort declarations must match"));
    }

    #[test]
    fn test_parse_simplify_basic() {
        let sexp = parse_sexp("(simplify (and true x))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::Simplify(Term::App(name, args)) => {
                assert_eq!(name, "and");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Simplify command"),
        }
    }

    #[test]
    fn test_parse_simplify_constant() {
        let sexp = parse_sexp("(simplify true)").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::Simplify(Term::Const(Constant::True)) => {}
            _ => panic!("Expected Simplify command with true constant"),
        }
    }

    #[test]
    fn test_parse_simplify_arithmetic() {
        let sexp = parse_sexp("(simplify (+ 1 2))").unwrap();
        let cmd = Command::from_sexp(&sexp).unwrap();
        match cmd {
            Command::Simplify(Term::App(name, args)) => {
                assert_eq!(name, "+");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Simplify command"),
        }
    }
}
