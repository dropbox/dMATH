* [Write rules][1]
* Write rules for Semgrep Code
* Rule structure syntax
On this page

* [Rule writing][2]

# Rule structure syntax

tip

Getting started with rule writing? Try the [Semgrep Tutorial][3] 🎓

This document describes the YAML rule syntax of Semgrep.

## Schema[​][4]

### Required[​][5]

All required fields must be present at the top level of a rule immediately under the `rules` key.

────┬───┬───────────────────────────────────────────────────────────────────────────────────────────
Fiel│Typ│Description                                                                                
d   │e  │                                                                                           
────┼───┼───────────────────────────────────────────────────────────────────────────────────────────
`id`│`st│Unique, descriptive identifier, for example: `no-unused-variable`                          
    │rin│                                                                                           
    │g` │                                                                                           
────┼───┼───────────────────────────────────────────────────────────────────────────────────────────
`mes│`st│Message that includes why Semgrep matched this pattern and how to remediate it. See also   
sage│rin│[Rule messages][6].                                                                        
`   │g` │                                                                                           
────┼───┼───────────────────────────────────────────────────────────────────────────────────────────
`sev│`st│Severity can be `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`. It indicates the criticality of    
erit│rin│issues detected by a rule. Note: Semgrep Supply Chain uses [CVE assignments for            
y`  │g` │severity][7], while the rule author sets severity for Code and Secrets. The older levels   
    │   │`ERROR`, `WARNING`, and `INFO` match `HIGH`, `MEDIUM`, and `LOW`. Severity values remain   
    │   │backwards compatible.                                                                      
────┼───┼───────────────────────────────────────────────────────────────────────────────────────────
`lan│`ar│See [language extensions and tags][8].                                                     
guag│ray│                                                                                           
es` │`  │                                                                                           
────┼───┼───────────────────────────────────────────────────────────────────────────────────────────
`pat│`st│Find code matching this expression                                                         
tern│rin│                                                                                           
`***│g` │                                                                                           
────┼───┼───────────────────────────────────────────────────────────────────────────────────────────
`pat│`ar│Logical `AND` of multiple patterns                                                         
tern│ray│                                                                                           
s`**│`  │                                                                                           
*   │   │                                                                                           
────┼───┼───────────────────────────────────────────────────────────────────────────────────────────
`pat│`ar│Logical `OR` of multiple patterns                                                          
tern│ray│                                                                                           
-eit│`  │                                                                                           
her`│   │                                                                                           
*** │   │                                                                                           
────┼───┼───────────────────────────────────────────────────────────────────────────────────────────
`pat│`st│Find code matching this [PCRE2][9]-compatible pattern in multiline mode                    
tern│rin│                                                                                           
-reg│g` │                                                                                           
ex`*│   │                                                                                           
**  │   │                                                                                           
────┴───┴───────────────────────────────────────────────────────────────────────────────────────────
info

Only one of the following keys are required: `pattern`, `patterns`, `pattern-either`,
`pattern-regex`

#### Language extensions and languages key values[​][10]

The following table includes languages supported by Semgrep, accepted file extensions for test files
that accompany the rules, and valid values that Semgrep rules require in the `languages` key.

─────────────────────────────────┬───────────────────────┬─────────────────────────
Language                         │Extensions             │`languages` key values   
─────────────────────────────────┼───────────────────────┼─────────────────────────
Apex (only in Semgrep Pro Engine)│`.cls`                 │`apex`                   
─────────────────────────────────┼───────────────────────┼─────────────────────────
Bash                             │`.bash`, `.sh`         │`bash`, `sh`             
─────────────────────────────────┼───────────────────────┼─────────────────────────
C                                │`.c`                   │`c`                      
─────────────────────────────────┼───────────────────────┼─────────────────────────
Cairo                            │`.cairo`               │`cairo`                  
─────────────────────────────────┼───────────────────────┼─────────────────────────
Clojure                          │`.clj`, `.cljs`,       │`clojure`                
                                 │`.cljc`, `.edn`        │                         
─────────────────────────────────┼───────────────────────┼─────────────────────────
C++                              │`.cc`, `.cpp`          │`cpp`, `c++`             
─────────────────────────────────┼───────────────────────┼─────────────────────────
C#                               │`.cs`                  │`csharp`, `c#`           
─────────────────────────────────┼───────────────────────┼─────────────────────────
Dart                             │`.dart`                │`dart`                   
─────────────────────────────────┼───────────────────────┼─────────────────────────
Dockerfile                       │`.dockerfile`,         │`dockerfile`, `docker`   
                                 │`.Dockerfile`          │                         
─────────────────────────────────┼───────────────────────┼─────────────────────────
Elixir                           │`.ex`, `.exs`          │`ex`, `elixir`           
─────────────────────────────────┼───────────────────────┼─────────────────────────
Generic                          │                       │`generic`                
─────────────────────────────────┼───────────────────────┼─────────────────────────
Go                               │`.go`                  │`go`, `golang`           
─────────────────────────────────┼───────────────────────┼─────────────────────────
HTML                             │`.htm`, `.html`        │`html`                   
─────────────────────────────────┼───────────────────────┼─────────────────────────
Java                             │`.java`                │`java`                   
─────────────────────────────────┼───────────────────────┼─────────────────────────
JavaScript                       │`.js`, `.jsx`          │`js`, `javascript`       
─────────────────────────────────┼───────────────────────┼─────────────────────────
JSON                             │`.json`, `.ipynb`      │`json`                   
─────────────────────────────────┼───────────────────────┼─────────────────────────
Jsonnet                          │`.jsonnet`,            │`jsonnet`                
                                 │`.libsonnet`           │                         
─────────────────────────────────┼───────────────────────┼─────────────────────────
JSX                              │`.js`, `.jsx`          │`js`, `javascript`       
─────────────────────────────────┼───────────────────────┼─────────────────────────
Julia                            │`.jl`                  │`julia`                  
─────────────────────────────────┼───────────────────────┼─────────────────────────
Kotlin                           │`.kt`, `.kts`, `.ktm`  │`kt`, `kotlin`           
─────────────────────────────────┼───────────────────────┼─────────────────────────
Lisp                             │`.lisp`, `.cl`, `.el`  │`lisp`                   
─────────────────────────────────┼───────────────────────┼─────────────────────────
Lua                              │`.lua`                 │`lua`                    
─────────────────────────────────┼───────────────────────┼─────────────────────────
OCaml                            │`.ml`, `.mli`          │`ocaml`                  
─────────────────────────────────┼───────────────────────┼─────────────────────────
PHP                              │`.php`, `.tpl`         │`php`                    
─────────────────────────────────┼───────────────────────┼─────────────────────────
Python                           │`.py`, `.pyi`          │`python`, `python2`,     
                                 │                       │`python3`, `py`          
─────────────────────────────────┼───────────────────────┼─────────────────────────
R                                │`.r`, `.R`             │`r`                      
─────────────────────────────────┼───────────────────────┼─────────────────────────
Ruby                             │`.rb`                  │`ruby`                   
─────────────────────────────────┼───────────────────────┼─────────────────────────
Rust                             │`.rs`                  │`rust`                   
─────────────────────────────────┼───────────────────────┼─────────────────────────
Scala                            │`.scala`               │`scala`                  
─────────────────────────────────┼───────────────────────┼─────────────────────────
Scheme                           │`.scm`, `.ss`          │`scheme`                 
─────────────────────────────────┼───────────────────────┼─────────────────────────
Solidity                         │`.sol`                 │`solidity`, `sol`        
─────────────────────────────────┼───────────────────────┼─────────────────────────
Swift                            │`.swift`               │`swift`                  
─────────────────────────────────┼───────────────────────┼─────────────────────────
Terraform                        │`.tf`, `.hcl`          │`tf`, `hcl`, `terraform` 
─────────────────────────────────┼───────────────────────┼─────────────────────────
TypeScript                       │`.ts`, `.tsx`          │`ts`, `typescript`       
─────────────────────────────────┼───────────────────────┼─────────────────────────
YAML                             │`.yml`, `.yaml`        │`yaml`                   
─────────────────────────────────┼───────────────────────┼─────────────────────────
XML                              │`.xml`                 │`xml`                    
─────────────────────────────────┴───────────────────────┴─────────────────────────
info

To see the maturity level of each supported language, see the following references:

* [Semgrep CE][11]
* [Semgrep Code][12]

### Optional[​][13]

──────────────┬─────┬───────────────────────────────────────────────────────────────────────────────
Field         │Type │Description                                                                    
──────────────┼─────┼───────────────────────────────────────────────────────────────────────────────
[`options`][14│`obje│Options object to turn on or turn off matching features                        
]             │ct`  │                                                                               
──────────────┼─────┼───────────────────────────────────────────────────────────────────────────────
[`fix`][15]   │`obje│Simple search-and-replace autofix capability                                   
              │ct`  │                                                                               
──────────────┼─────┼───────────────────────────────────────────────────────────────────────────────
[`metadata`][1│`obje│Arbitrary user-provided data; attach data to rules without affecting Semgrep   
6]            │ct`  │behavior                                                                       
──────────────┼─────┼───────────────────────────────────────────────────────────────────────────────
[`min-version`│`stri│Minimum Semgrep version compatible with the rule                               
][17]         │ng`  │                                                                               
──────────────┼─────┼───────────────────────────────────────────────────────────────────────────────
[`max-version`│`stri│Maximum Semgrep version compatible with the rule                               
][18]         │ng`  │                                                                               
──────────────┼─────┼───────────────────────────────────────────────────────────────────────────────
[`paths`][19] │`obje│Paths to include or exclude when running the rule                              
              │ct`  │                                                                               
──────────────┴─────┴───────────────────────────────────────────────────────────────────────────────

The following field is optional, but if used, it must be nested underneath a `patterns` or
`pattern-either` field.

───────────────────┬──────┬──────────────────────────────────────────
Field              │Type  │Description                               
───────────────────┼──────┼──────────────────────────────────────────
[`pattern-inside`][│`strin│Keep findings that lie inside this pattern
20]                │g`    │                                          
───────────────────┴──────┴──────────────────────────────────────────

The following fields are optional, but if used, they must be nested underneath a `patterns` field.

──────────────────────┬────┬────────────────────────────────────────────────────────────────────────
Field                 │Type│Description                                                             
──────────────────────┼────┼────────────────────────────────────────────────────────────────────────
[`metavariable-regex`]│`map│Search metavariables for [Python `re`][22] compatible expressions; regex
[21]                  │`   │matching is **left anchored**                                           
──────────────────────┼────┼────────────────────────────────────────────────────────────────────────
[`metavariable-pattern│`map│Match metavariables with a pattern formula                              
`][23]                │`   │                                                                        
──────────────────────┼────┼────────────────────────────────────────────────────────────────────────
[`metavariable-compari│`map│Compare metavariables against basic [Python expressions][25]            
son`][24]             │`   │                                                                        
──────────────────────┼────┼────────────────────────────────────────────────────────────────────────
[`metavariable-name`][│`map│Match metavariables against constraints on what they name               
26]                   │`   │                                                                        
──────────────────────┼────┼────────────────────────────────────────────────────────────────────────
[`pattern-not`][27]   │`str│Logical `NOT` - remove findings matching this expression                
                      │ing`│                                                                        
──────────────────────┼────┼────────────────────────────────────────────────────────────────────────
[`pattern-not-inside`]│`str│Keep findings that do not lie inside this pattern                       
[28]                  │ing`│                                                                        
──────────────────────┼────┼────────────────────────────────────────────────────────────────────────
[`pattern-not-regex`][│`str│Filter results using a [PCRE2][30]-compatible pattern in multiline mode 
29]                   │ing`│                                                                        
──────────────────────┴────┴────────────────────────────────────────────────────────────────────────

## Operators[​][31]

### `pattern`[​][32]

The `pattern` operator looks for code matching its expression. This can be basic expressions like
`$X == $X` or unwanted function calls like `hashlib.md5(...)`.

`rules:
  - id: md5-usage
    languages:
      - python
    message: Found md5 usage
    pattern: hashlib.md5(...)
    severity: HIGH
`

The preceding pattern matches the following:

`import hashlib
# ruleid: md5-usage
digest = hashlib.md5(b"test")
# ok: md5-usage
digest = hashlib.sha256(b"test")
`

### `patterns`[​][33]

The `patterns` operator performs a logical `AND` operation on one or more child patterns. This is
useful for chaining multiple patterns together where all patterns must be true.

`rules:
  - id: unverified-db-query
    patterns:
      - pattern: db_query(...)
      - pattern-not: db_query(..., verify=True, ...)
    message: Found unverified db query
    severity: HIGH
    languages:
      - python
`

The preceding pattern matches the following:

`# ruleid: unverified-db-query
db_query("SELECT * FROM ...")
# ok: unverified-db-query
db_query("SELECT * FROM ...", verify=True, env="prod")
`

#### `patterns` operator evaluation strategy[​][34]

The order in which the child patterns are declared in a `patterns` operator does not affect the
final result. A `patterns` operator is always evaluated in the same way:

1. Semgrep evaluates all *positive* patterns, including [`pattern-inside`][35]s, [`pattern`][36]s,
   [`pattern-regex`][37]es, and [`pattern-either`][38]s. Each range matched by one of these patterns
   is intersected with the ranges matched by the other operators. The result is a set of *positive*
   ranges. The positive ranges carry *metavariable bindings*. For example, in one range,`$X` can be
   bound to the function call `foo()`, and in another range `$X` can be bound to the expression `a +
   b`.
2. Semgrep evaluates all *negative* patterns, including [`pattern-not-inside`][39]s,
   [`pattern-not`][40]s, and [`pattern-not-regex`][41]es. This provides a set of *negative ranges*
   which are used to filter the positive ranges. This results in a strict subset of the positive
   ranges computed in the previous step.
3. Semgrep evaluates all *conditionals*, including [`metavariable-regex`][42]es,
   [`metavariable-pattern`][43]s, and [`metavariable-comparison`][44]s. These conditional operators
   can only examine the metavariables bound in the positive ranges in step 1 and have been filtered
   through the negative patterns in step 2. Note that metavariables bound by negative patterns are
   *not* available here.
4. Semgrep applies all [`focus-metavariable`][45]s by computing the intersection of each positive
   range with the range of the metavariable on which you want to focus. Again, the only
   metavariables available to focus on are those bound by positive patterns.

### `pattern-either`[​][46]

The `pattern-either` operator performs a logical `OR` operation on one or more child patterns. This
is useful for chaining multiple patterns together where any may be true.

`rules:
  - id: insecure-crypto-usage
    pattern-either:
      - pattern: hashlib.sha1(...)
      - pattern: hashlib.md5(...)
    message: Found insecure crypto usage
    languages:
      - python
    severity: HIGH
`

The preceding pattern matches the following:

`import hashlib
# ruleid: insecure-crypto-usage
digest = hashlib.md5(b"test")
# ruleid: insecure-crypto-usage
digest = hashlib.sha1(b"test")
# ok: insecure-crypto-usage
digest = hashlib.sha256(b"test")
`

This rule checks for the use of Python standard library functions `hashlib.md5` or `hashlib.sha1`.
Depending on their usage, these hashing functions are [considered insecure][47].

### `pattern-regex`[​][48]

The `pattern-regex` operator searches files for substrings matching the given [Perl-Compatible
Regular Expressions (PCRE)][49] pattern. PCRE is a full-featured regular expression (regex) library
that is widely compatible with Perl, as well as with the respective regex libraries of Python,
JavaScript, Go, Ruby, and Java. This is useful for migrating existing regular expression code search
capability to Semgrep. Patterns are compiled in multiline mode. For example, `^` and `$` match at
the beginning and end of lines, respectively, in addition to the beginning and end of input.

caution

PCRE2 supports [some Unicode character properties, but not some Perl properties][50]. For example,
`\p{Egyptian_Hieroglyphs}` is supported, but `\p{InMusicalSymbols}` isn't.

#### Example: `pattern-regex` combined with other pattern operators[​][51]

`rules:
  - id: boto-client-ip
    patterns:
      - pattern-inside: boto3.client(host="...")
      - pattern-regex: \d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}
    message: boto client using IP address
    languages:
      - python
    severity: HIGH
`

The preceding pattern matches the following:

`import boto3
# ruleid: boto-client-ip
client = boto3.client(host="192.168.1.200")
# ok: boto-client-ip
client = boto3.client(host="dev.internal.example.com")
`

#### Example: `pattern-regex` used as a standalone, top-level operator[​][52]

`rules:
  - id: legacy-eval-search
    pattern-regex: eval\(
    message: Insecure code execution
    languages:
      - javascript
    severity: HIGH
`

The preceding pattern matches the following:

`# ruleid: legacy-eval-search
eval('var a = 5')
`
info

Single (`'`) and double (`"`) quotes [behave differently][53] in YAML syntax. Single quotes are
typically preferred when using backslashes (`\`) with `pattern-regex`.

Note that you may bind a section of a regular expression to a metavariable by using [named capturing
groups][54]. In this case, the name of the capturing group must be a valid metavariable name.

`rules:
  - id: my_pattern_id-copy
    patterns:
      - pattern-regex: a(?P<FIRST>.*)b(?P<SECOND>.*)
    message: Semgrep found a match, with $FIRST and $SECOND
    languages:
      - regex
    severity: MEDIUM
`

The preceding pattern matches the following:

`acbd
`

### `pattern-not-regex`[​][55]

The `pattern-not-regex` operator filters results using a [PCRE2][56] regular expression in multiline
mode. This is most useful when combined with regular-expression-only rules, providing an easy way to
filter findings without having to use negative lookaheads. `pattern-not-regex` works with regular
`pattern` clauses, too.

The syntax for this operator is the same as `pattern-regex`.

This operator filters findings that have *any overlap* with the supplied regular expression. For
example, if you use `pattern-regex` to detect `Foo==1.1.1` and it also detects `Foo-Bar==3.0.8` and
`Bar-Foo==3.0.8`, you can use `pattern-not-regex` to filter the unwanted findings.

`rules:
  - id: detect-only-foo-package
    languages:
      - regex
    message: Found foo package
    patterns:
      - pattern-regex: foo
      - pattern-not-regex: foo-
      - pattern-not-regex: -foo
    severity: HIGH
`

The preceding pattern matches the following:

`# ruleid: detect-only-foo-package
foo==1.1.1
# ok: detect-only-foo-package
foo-bar==3.0.8
# ok: detect-only-foo-package
bar-foo==3.0.8
`

### `focus-metavariable`[​][57]

The `focus-metavariable` operator focuses on, or *zooms in* on, the code region matched by a single
metavariable or a list of metavariables. For example, to find all functions' arguments annotated
with the type `bad`, you may write the following pattern:

`pattern: |
  def $FUNC(..., $ARG : bad, ...):
    ...
`

This works, but it matches the entire function definition. Sometimes, this is not desirable. If the
definition spans hundreds of lines, they are all matched. In particular, if you are using [Semgrep
AppSec Platform][58] and you have triaged a finding generated by this pattern, the same finding
shows up again as new if you make any change to the definition of the function!

To specify that you are only interested in the code matched by a particular metavariable, which, in
the example, is `$ARG`, use `focus-metavariable`.

`rules:
  - id: find-bad-args
    patterns:
      - pattern: |
          def $FUNC(..., $ARG : bad, ...):
            ...
      - focus-metavariable: $ARG
    message: |
      `$ARG' has a "bad" type!
    languages:
      - python
    severity: MEDIUM
`

The preceding pattern matches the following:

`def f(x : bad):
    return x
`

Note that `focus-metavariable: $ARG` is not the same as `pattern: $ARG`! Using `pattern: $ARG` finds
all the uses of the parameter `x`, which is not the desired behavior! (Note that `pattern: $ARG`
does not match the formal parameter declaration, because in this context `$ARG` only matches
expressions.)

`rules:
  - id: find-bad-args
    patterns:
      - pattern: |
          def $FUNC(..., $ARG : bad, ...):
            ...
      - pattern: $ARG
    message: |
      `$ARG' has a "bad" type!
    languages:
      - python
    severity: MEDIUM
`

The preceding pattern matches the following:

`def f(x : bad):
    return x
`

In short, `focus-metavariable: $X` is not a pattern in itself. It does not perform any matching; it
only focuses the matching on the code already bound to `$X` by other patterns. On the other hand,
`pattern: $X` matches `$X` against your code (and in this context, `$X` only matches expressions)!

#### Including multiple focus metavariables using set intersection semantics[​][59]

Include more `focus-metavariable` keys with different metavariables under the `pattern` to match
results **only** for the overlapping region of all the focused code:

`    patterns:
      - pattern: foo($X, ..., $Y)
      - focus-metavariable:
        - $X
        - $Y
`
`rules:
  - id: intersect-focus-metavariable
    patterns:
      - pattern-inside: foo($X, ...)
      - focus-metavariable: $X
      - pattern: $Y + ...
      - focus-metavariable: $Y
      - pattern: "1"
    message: Like set intersection, only the overlapping region is highlighted
    languages:
      - python
    severity: HIGH
`

The preceding pattern matches the following:

`# ruleid: intersect-focus-metavariable
foo (
    1
    +
    2,
    1
)

# OK: test
foo (2+ 1, 1)
`
info

To make a list of multiple focus metavariables using set union semantics that matches the
metavariables regardless of their position in code, see [Including multiple focus metavariables
using set union semantics][60] documentation.

### `metavariable-regex`[​][61]

The `metavariable-regex` operator searches metavariables for a [PCRE2][62] regular expression. This
is useful for filtering results based on a [metavariable’s][63] value. It requires the
`metavariable` and `regex` keys and can be combined with other pattern operators.

`rules:
  - id: insecure-methods
    patterns:
      - pattern: module.$METHOD(...)
      - metavariable-regex:
          metavariable: $METHOD
          regex: (insecure)
    message: module using insecure method call
    languages:
      - python
    severity: HIGH
`

The preceding pattern matches the following:

`# ruleid: insecure-methods
module.insecure1("test")
# ruleid: insecure-methods
module.insecure2("test")
# ruleid: insecure-methods
module.insecure3("test")
# ok: insecure-methods
module.secure("test")
`

Regex matching is **left anchored**. To allow prefixes, use `.*` at the beginning of the regex. To
match the end of a string, use `$`. The following example, using the same expression as above but
anchored on the right, finds no matches:

`rules:
  - id: insecure-methods
    patterns:
      - pattern: module.$METHOD(...)
      - metavariable-regex:
          metavariable: $METHOD
          regex: (insecure$)
    message: module using insecure method call
    languages:
      - python
    severity: HIGH

The following example matches all of the function calls in the same code sample, returning a false p
ositive on the `module.secure` call:

```yaml
rules:
  - id: insecure-methods
    patterns:
      - pattern: module.$METHOD(...)
      - metavariable-regex:
          metavariable: $METHOD
          regex: (.*secure)
    message: module using insecure method call
    languages:
      - python
    severity: HIGH
`
info

Include quotes in your regular expression when using `metavariable-regex` to search string literals.
For more details, see [include-quotes][64] code snippet.

### `metavariable-pattern`[​][65]

The `metavariable-pattern` operator matches metavariables with a pattern formula. This is useful for
filtering results based on a [metavariable’s][66] value. It requires the `metavariable` key, and
precisely one key of `pattern`, `patterns`, `pattern-either`, or `pattern-regex`. This operator can
be nested as well as combined with other operators.

For example, the `metavariable-pattern` can be used to filter out matches that do **not** match
specific criteria:

`rules:
  - id: disallow-old-tls-versions2
    languages:
      - javascript
    message: Match found
    patterns:
      - pattern: |
          $CONST = require('crypto');
          ...
          $OPTIONS = $OPTS;
          ...
          https.createServer($OPTIONS, ...);
      - metavariable-pattern:
          metavariable: $OPTS
          patterns:
            - pattern-not: >
                {secureOptions: $CONST.SSL_OP_NO_SSLv2 | $CONST.SSL_OP_NO_SSLv3
                | $CONST.SSL_OP_NO_TLSv1}
    severity: MEDIUM
`

The preceding pattern matches the following:

`function bad() {
    // ruleid:disallow-old-tls-versions2
    var constants = require('crypto');
    var sslOptions = {
    key: fs.readFileSync('/etc/ssl/private/private.key'),
    secureProtocol: 'SSLv23_server_method',
    secureOptions: constants.SSL_OP_NO_SSLv2 | constants.SSL_OP_NO_SSLv3
    };
    https.createServer(sslOptions);
}
`
info

In this case, it is possible to start a `patterns` AND operation with a `pattern-not`, because there
is an implicit `pattern: ...` that matches the content of the metavariable.

The `metavariable-pattern` is also helpful in combination with `pattern-either`:

`rules:
  - id: open-redirect
    languages:
      - python
    message: Match found
    patterns:
      - pattern-inside: |
          def $FUNC(...):
            ...
            return django.http.HttpResponseRedirect(..., $DATA, ...)
      - metavariable-pattern:
          metavariable: $DATA
          patterns:
            - pattern-either:
                - pattern: $REQUEST
                - pattern: $STR.format(..., $REQUEST, ...)
                - pattern: $STR % $REQUEST
                - pattern: $STR + $REQUEST
                - pattern: f"...{$REQUEST}..."
            - metavariable-pattern:
                metavariable: $REQUEST
                patterns:
                  - pattern-either:
                      - pattern: request.$W
                      - pattern: request.$W.get(...)
                      - pattern: request.$W(...)
                      - pattern: request.$W[...]
                  - metavariable-regex:
                      metavariable: $W
                      regex: (?!get_full_path)
    severity: MEDIUM
`

The preceding pattern matches the following:

`from django.http import HttpResponseRedirect
def unsafe(request):
    # ruleid:open-redirect
    return HttpResponseRedirect(request.POST.get("url"))
`
tip

It is possible to nest `metavariable-pattern` inside `metavariable-pattern`!

info

The metavariable should be bound to an expression, a statement, or a list of statements, for this
test to be meaningful. A metavariable bound to a list of function arguments, a type, or a pattern
always evaluates to false.

#### `metavariable-pattern` with nested language[​][67]

If the metavariable's content is a string, then it is possible to use `metavariable-pattern` to
match this string as code by specifying the target language via the `language` key. See the
following examples of `metavariable-pattern`:

Examples of `metavariable-pattern`

* Match JavaScript code inside HTML in the following [Semgrep Playground][68] example.
* Filter regex matches in the following [Semgrep Playground][69] example.

#### Example: Match JavaScript code inside HTML[​][70]

`rules:
  - id: test
    languages:
      - generic
    message: javascript inside html working!
    patterns:
      - pattern: |
          <script ...>$...JS</script>
      - metavariable-pattern:
          language: javascript
          metavariable: $...JS
          patterns:
            - pattern: |
                console.log(...)
    severity: MEDIUM
`

The preceding pattern matches the following:

`<!-- ruleid:test -->
<script>
console.log("hello")
</script>
`

#### Example: Filter regex matches[​][71]

`rules:
  - id: test
    languages:
      - generic
    message: "Google dependency: $1 $2"
    patterns:
      - pattern-regex: gem "(.*)", "(.*)"
      - metavariable-pattern:
          metavariable: $1
          language: generic
          patterns:
            - pattern: google
    severity: LOW
`

The preceding pattern matches the following:

`source "https://rubygems.org"

#OK:test
gem "functions_framework", "~> 0.7"
#ruleid:test
gem "google-cloud-storage", "~> 1.29"
`

### `metavariable-comparison`[​][72]

The `metavariable-comparison` operator compares metavariables against a basic [Python
comparison][73] expression. This is useful for filtering results based on a [metavariable's][74]
numeric value.

The `metavariable-comparison` operator is a mapping that requires the `metavariable` and
`comparison` keys. It can be combined with other pattern operators in the following [Semgrep
Playground][75] example.

This matches code such as `set_port(80)` or `set_port(443)`, but not `set_port(8080)`.

Comparison expressions support simple arithmetic as well as composition with [Boolean operators][76]
to allow for more complex matching. This is particularly useful for checking that metavariables are
divisible by particular values, such as enforcing that a specific value is even or odd.

`rules:
  - id: superuser-port
    languages:
      - python
    message: module setting superuser port
    patterns:
      - pattern: set_port($ARG)
      - metavariable-comparison:
          comparison: $ARG < 1024 and $ARG % 2 == 0
          metavariable: $ARG
    severity: HIGH
`

The preceding pattern matches the following:

`# ok: superuser-port
set_port(443)
# ruleid: superuser-port
set_port(80)
# ok: superuser-port
set_port(8080)
`

Building on the previous example, this still matches code such as `set_port(80)`, but it no longer
matches `set_port(443)` or `set_port(8080)`.

The `comparison` key accepts a Python expression using:

* Boolean, string, integer, and float literals.
* Boolean operators `not`, `or`, and `and`.
* Arithmetic operators `+`, `-`, `*`, `/`, and `%`.
* Comparison operators `==`, `!=`, `<`, `<=`, `>`, and `>=`.
* Function `int()` to convert strings into integers.
* Function `str()` to convert numbers into strings.
* Function `today()` that gets today's date as a float representing epoch time.
* Function `strptime()` that converts strings in the format `"yyyy-mm-dd"` to a float representing
  the date in epoch time.
* Lists, together with the `in`, and `not in` infix operators.
* Strings, together with the `in` and `not in` infix operators, for substring containment.
* Function `re.match()` to match a regular expression (without the optional `flags` argument).

You can use Semgrep metavariables such as `$MVAR`, which Semgrep evaluates as follows:

* If `$MVAR` binds to a literal, then that literal is the value assigned to `$MVAR`.
* If `$MVAR` binds to a code variable that is a constant, and constant propagation is enabled (as it
  is by default), then that constant is the value assigned to `$MVAR`.
* Otherwise, the code bound to the `$MVAR` is kept unevaluated, and its string representation can be
  obtained using the `str()` function, as in `str($MVAR)`. For example, if `$MVAR` binds to the code
  variable `x`, `str($MVAR)` evaluates to the string literal `"x"`.

#### Legacy `metavariable-comparison` keys[​][77]

info

You can avoid using the legacy keys described below (`base: int` and `strip: bool`) by using the
`int()` function, as in `int($ARG) > 0o600` or `int($ARG) > 2147483647`.

The `metavariable-comparison` operator also takes optional `base: int` and `strip: bool` keys. These
keys set the integer base the metavariable value should be interpreted as and remove quotes from the
metavariable value, respectively.

`rules:
  - id: excessive-permissions
    languages:
      - python
    message: module setting excessive permissions
    patterns:
      - pattern: set_permissions($ARG)
      - metavariable-comparison:
          comparison: $ARG > 0o600
          metavariable: $ARG
          base: 8
    severity: HIGH
`

The preceding pattern matches the following:

`# ruleid: excessive-permissions
set_permissions(0o700)
# ok: excessive-permissions
set_permissions(0o400)
`

This interprets metavariable values found in code as octal. As a result, Semgrep detects `0700`, but
it does **not** detect `0400`.

`rules:
  - id: int-overflow
    languages:
      - python
    message: Potential integer overflow
    patterns:
      - pattern: int($ARG)
      - metavariable-comparison:
          strip: true
          comparison: $ARG > 2147483647
          metavariable: $ARG
    severity: HIGH
`

The preceding pattern matches the following:

`# ruleid: int-overflow
int("2147483648")
# ok: int-overflow
int("2147483646")
`

This removes quotes (`'`, `"`, and ```) from both ends of the metavariable content. As a result,
Semgrep detects `"2147483648"`, but it does **not** detect `"2147483646"`. This is useful when you
expect strings to contain integer or float data.

### `metavariable-name`[​][78]

tip

* `metavariable-name` requires a Semgrep account and the use of Semgrep's proprietary engine since
  it requires name resolution information. This means that it does **not** work with the
  `--oss-only` flag.
* While optional, you can improve the accuracy of `metavariable-name` by enabling **[cross-file
  analysis][79]**.

The `metavariable-name` operator adds a constraint to the types of identifiers a metavariable can
match. Currently, the only constraint supported is on the module or namespace from which an
identifier originates. This is useful for filtering results in languages that don't have a native
syntax for fully qualified names, or languages where module names may contain characters that are
not legal in identifiers, such as JavaScript or TypeScript.

`rules:
  - id: insecure-method
    patterns:
      - pattern: $MODULE.insecure(...)
      - metavariable-name:
          metavariable: $MODULE
          module: "@foo-bar"
    message: Uses insecure method from @foo-bar.
    languages:
      - javascript
    severity: HIGH
`

The preceding pattern matches the following:

`// ECMAScript modules
import * as lib from '@foo-bar';
import * as lib2 from 'myotherlib';

// CommonJS modules
const { insecure } = require('@foo-bar');
const lib3 = require('myotherlib');

// ruleid: insecure-method
lib.insecure("test");
// ruleid: insecure-method
insecure("test");

// ok: insecure-method
lib.secure("test");
// ok: insecure-method
lib2.insecure("test");
// ok: insecure-method
lib3.insecure("test");
`

If a match should occur if the metavariable matches one of a variety of matches, there is also a
shorthand `modules` key, which takes a list of module names.

`rules:
  - id: insecure-method
    patterns:
      - pattern: $MODULE.method(...)
      - metavariable-regex:
          metavariable: $MODULE
          modules:
           - foo
           - bar
    message: Uses insecure method from @foo-bar.
    languages:
      - javascript
    severity: HIGH
`

This can be useful in instances where there may be multiple API-compatible packages that share an
issue.

### `pattern-not`[​][80]

The `pattern-not` operator is the opposite of the `pattern` operator. It finds code that does not
match its expression. This is useful for eliminating common false positives.

`rules:
  - id: unverified-db-query
    patterns:
      - pattern: db_query(...)
      - pattern-not: db_query(..., verify=True, ...)
    message: Found unverified db query
    severity: HIGH
    languages:
      - python
`

The preceding pattern matches the following:

`# ruleid: unverified-db-query
db_query("SELECT * FROM ...")
# ok: unverified-db-query
db_query("SELECT * FROM ...", verify=True, env="prod")
`

Alternatively, `pattern-not` accepts a `patterns` or `pattern-either` property and negates
everything inside the property.

`rules:
  - id: unverified-db-query
    patterns:
      - pattern: db_query(...)
      - pattern-not:
          pattern-either:
            - pattern: db_query(..., verify=True, ...)
            - pattern-inside: |
                with ensure_verified(db_query):
                  db_query(...)
    message: Found unverified db query
    severity: HIGH
    languages:
      - python
`

### `pattern-inside`[​][81]

The `pattern-inside` operator keeps matched findings that reside within its expression. This is
useful for finding code within other pieces of code, such as functions or if blocks.

`rules:
  - id: return-in-init
    patterns:
      - pattern: return ...
      - pattern-inside: |
          class $CLASS:
            ...
      - pattern-inside: |
          def __init__(...):
              ...
    message: return should never appear inside a class __init__ function
    languages:
      - python
    severity: HIGH
`

The preceding pattern matches the following:

`class A:
    def __init__(self):
        # ruleid: return-in-init
        return None

class B:
    def __init__(self):
        # ok: return-in-init
        self.inited = True

def foo():
    # ok: return-in-init
    return 5
`

### `pattern-not-inside`[​][82]

The `pattern-not-inside` operator keeps matched findings that do not reside within its expression.
It is the opposite of `pattern-inside`. This is useful for finding code that’s missing a
corresponding cleanup action like disconnect, close, or shutdown. It’s also helpful in finding
problematic code that isn't inside code that mitigates the issue.

`rules:
  - id: open-never-closed
    patterns:
      - pattern: $F = open(...)
      - pattern-not-inside: |
          $F = open(...)
          ...
          $F.close()
    message: file object opened without a corresponding close
    languages:
      - python
    severity: HIGH
`

The preceding pattern matches the following:

`def func1():
    # ruleid: open-never-closed
    fd = open('test.txt')
    results = fd.read()
    return results
def func2():
    # ok: open-never-closed
    fd = open('test.txt')
    results = fd.read()
    fd.close()
    return results
`

The preceding rule identifies files that are opened but never closed, potentially leading to
resource exhaustion. It looks for the `open(...)` pattern *and not* a following `close()` pattern.

The `$F` metavariable ensures that the same variable name is used in the `open` and `close` calls.
The ellipsis operator allows any arguments to be passed to `open` and any sequence of code
statements to be executed between the `open` and `close` calls. The rule ignores how `open` is
called or what happens up to a `close` call; it only needs to make sure `close` is called.

## Metavariable matches[​][83]

Metavariable matching operates differently for logical AND (`patterns`) and logical OR
(`pattern-either`) parent operators. Behavior is consistent across all child operators: `pattern`,
`pattern-not`, `pattern-regex`, `pattern-inside`, `pattern-not-inside`.

### Metavariables in logical ANDs[​][84]

Metavariable values must be identical across sub-patterns when performing logical AND operations
with the `patterns` operator.

Example:

`rules:
  - id: function-args-to-open
    patterns:
      - pattern-inside: |
          def $F($X):
              ...
      - pattern: open($X)
    message: "Function argument passed to open() builtin"
    languages: [python]
    severity: HIGH
`

This rule matches the following code:

`def foo(path):
    open(path)
`

The example rule doesn’t match this code:

`def foo(path):
    open(something_else)
`

### Metavariables in logical ORs[​][85]

Metavariable matching does not affect the matching of logical OR operations with the
`pattern-either` operator.

Example:

`rules:
   - id: insecure-function-call
    pattern-either:
      - pattern: insecure_func1($X)
      - pattern: insecure_func2($X)
    message: "Insecure function use"
    languages: [python]
    severity: HIGH
`

The preceding rule matches both examples below:

`insecure_func1(something)
insecure_func2(something)
`
`insecure_func1(something)
insecure_func2(something_else)
`

### Metavariables in complex logic[​][86]

Metavariable matching still affects subsequent logical ORs if the parent is a logical AND.

Example:

`patterns:
  - pattern-inside: |
      def $F($X):
        ...
  - pattern-either:
      - pattern: bar($X)
      - pattern: baz($X)
`

The preceding rule matches both examples below:

`def foo(something):
   bar(something)
`
`def foo(something):
   baz(something)
`

The example rule doesn’t match this code:

`def foo(something):
   bar(something_else)
`

## `options`[​][87]

Enable, disable, or modify the following matching features:

───┬───┬────────────────────────────────────────────────────────────────────────────────────────────
Opt│Def│Description                                                                                 
ion│aul│                                                                                            
   │t  │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`ac│`tr│[Matching modulo associativity and commutativity][88], treat Boolean AND/OR as associative, 
_ma│ue`│and bitwise AND/OR/XOR as both associative and commutative.                                 
tch│   │                                                                                            
ing│   │                                                                                            
`  │   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`at│`tr│Expression patterns (for example: `f($X)`) matches attributes (for example: `@f(a)`).       
tr_│ue`│                                                                                            
exp│   │                                                                                            
r` │   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`co│`fa│Treat Boolean AND/OR as commutative even if not semantically accurate.                      
mmu│lse│                                                                                            
tat│`  │                                                                                            
ive│   │                                                                                            
_bo│   │                                                                                            
olo│   │                                                                                            
p` │   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`co│`tr│[Constant propagation][89], including [intraprocedural flow-sensitive constant              
nst│ue`│propagation][90].                                                                           
ant│   │                                                                                            
_pr│   │                                                                                            
opa│   │                                                                                            
gat│   │                                                                                            
ion│   │                                                                                            
`  │   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`de│`fa│Match non-keyword attributes (for example: decorators in Python) in order, instead of the   
cor│lse│order-agnostic default. Keyword attributes (for example: `static`, `inline`, etc) are not   
ato│`  │affected.                                                                                   
rs_│   │                                                                                            
ord│   │                                                                                            
er_│   │                                                                                            
mat│   │                                                                                            
ter│   │                                                                                            
s` │   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`ge│non│In generic mode, assume that comments follow the specified syntax. They are then ignored for
ner│e  │matching purposes. Allowed values for comment styles are:                                   
ic_│   │                                                                                            
com│   │* `c` for traditional C-style comments (`/* ... */`).                                       
men│   │* `cpp` for modern C or C++ comments (`// ...` or `/* ... */`).                             
t_s│   │* `shell` for shell-style comments (`# ...`).                                               
tyl│   │By default, the generic mode does not recognize any comments. Available since Semgrep       
e` │   │version 0.96. For more information about generic mode, see the [Generic Pattern             
   │   │Matching][91] documentation.                                                                
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`ge│`10│In generic mode, this is the maximum number of newlines that an ellipsis operator `...` can 
ner│`  │match, or equivalently, the maximum number of lines covered by the match minus one. The     
ic_│   │default value is `10` (newlines) for performance reasons. Increase it with caution. Note    
ell│   │that the same effect as `20` can be achieved without changing this setting and by writing   
ips│   │`... ...` in the pattern instead of `...`. Setting it to `0` is useful with line-oriented   
is_│   │languages (for example, [INI][92] or key-value pairs in general) to prevent a match from    
max│   │extending to the next line of code. Available since Semgrep 0.96. For more information about
_sp│   │generic mode, see [Generic pattern matching][93] documentation.                             
an`│   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`im│`tr│Return statement patterns (for example `return $E`) match expressions that may be evaluated 
pli│ue`│last in a function as if there was a return keyword in front of those expressions. Only     
cit│   │applies to certain expression-based languages, such as Ruby and Julia.                      
_re│   │                                                                                            
tur│   │                                                                                            
n` │   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`in│`fa│Set this value to `true` for Semgrep to run this rule with cross-function and cross-file    
ter│lse│analysis. It is **required** for rules that use cross-function, cross-file analysis.        
fil│`  │                                                                                            
e` │   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`sy│`fa│Treat equal operations as symmetric (for example: `a == b` is equal to `b == a`).           
mme│lse│                                                                                            
tri│`  │                                                                                            
c_e│   │                                                                                            
q` │   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`ta│`fa│Experimental option which are be subject to future changes. Used in taint analysis. Assume  
int│lse│that function calls do **not** propagate taint from their arguments to their output.        
_as│`  │Otherwise, Semgrep always assumes that functions may propagate taint. Can replace           
sum│   │**not-conflicting** sanitizers added in v0.69.0 in the future.                              
e_s│   │                                                                                            
afe│   │                                                                                            
_fu│   │                                                                                            
nct│   │                                                                                            
ion│   │                                                                                            
s` │   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`ta│`fa│Used in taint analysis. Assume that an array-access expression is safe even if the index    
int│lse│expression is tainted. Otherwise, Semgrep assumes that, for example, `a[i]` is tainted if `i
_as│`  │is tainted, even if `a` is not. Enabling this option is recommended for high-signal rules,  
sum│   │whereas disabling it is preferred for audit rules. Currently, it is disabled by default to  
e_s│   │maintain backward compatibility, but this may change in the near future after further       
afe│   │evaluation.                                                                                 
_in│   │                                                                                            
dex│   │                                                                                            
es`│   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`va│`tr│Assignment patterns (for example `$X = $E`) match variable declarations (for example `var x 
rde│ue`│= 1;`).                                                                                     
f_a│   │                                                                                            
ssi│   │                                                                                            
gn`│   │                                                                                            
───┼───┼────────────────────────────────────────────────────────────────────────────────────────────
`xm│`tr│Any XML/JSX/HTML element patterns have implicit ellipsis for attributes (for example: `<div 
l_a│ue`│/>` matches `<div foo="1">`.                                                                
ttr│   │                                                                                            
s_i│   │                                                                                            
mpl│   │                                                                                            
ici│   │                                                                                            
t_e│   │                                                                                            
lli│   │                                                                                            
psi│   │                                                                                            
s` │   │                                                                                            
───┴───┴────────────────────────────────────────────────────────────────────────────────────────────

The complete list of available options can be consulted in the [Semgrep matching engine
configuration][94] module. Please note that options not included in the table above are considered
experimental and may change or be removed without notice.

## `fix`[​][95]

The `fix` top-level key allows for simple autofixing of a pattern by suggesting an autofix for each
match. Run `semgrep` with `--autofix` to apply the changes to the files.

Example:

`rules:
  - id: use-dict-get
    patterns:
      - pattern: $DICT[$KEY]
    fix: $DICT.get($KEY)
    message: "Use `.get()` method to avoid a KeyNotFound error"
    languages: [python]
    severity: HIGH
`

For more information about `fix` and `--autofix` see [Autofix][96] documentation.

## `metadata`[​][97]

Provide additional information for a rule with the `metadata:` key, such as a related CWE,
likelihood, or OWASP.

Example:

`rules:
  - id: eqeq-is-bad
    patterns:
      - [...]
    message: "useless comparison operation `$X == $X` or `$X != $X`"
    metadata:
      cve: CVE-2077-1234
      discovered-by: Ikwa L'equale
    languages:
      - javascript
      - python
      - go
    severity: MEDIUM
`

The metadata are also displayed in the output of Semgrep if you’re running it with `--json`. Rules
with `category: security` have additional metadata requirements. See [Including fields required by
security category][98] for more information.

## `min-version` and `max-version`[​][99]

Each rule supports optional fields `min-version` and `max-version` specifying minimum and maximum
Semgrep versions. If the Semgrep version being used doesn't satisfy these constraints, the rule is
skipped without causing a fatal error.

Example rule:

`rules:
  - id: bad-goflags
    # earlier semgrep versions can't parse the pattern
    min-version: 1.31.0
    pattern: |
      ENV ... GOFLAGS='-tags=dynamic -buildvcs=false' ...
    languages: [dockerfile]
    message: "We should not use these flags"
    severity: MEDIUM
`

Another use case is when a newer version of a rule works better than before but relies on a new
feature. In this case, you can use `min-version` and `max-version` to ensure that either the older
or the newer rule is used, but not both. The rules would look like this:

`rules:
  - id: something-wrong-v1
    max-version: 1.72.999
    ...
  - id: something-wrong-v2
    min-version: 1.73.0
    # 10x faster than v1!
    ...
`

The `min-version`/`max-version` feature has been available since Semgrep 1.38.0. It is intended
primarily for publishing rules that rely on newly released features without causing errors in older
Semgrep installations.

## `category`[​][100]

Provide a category for users of the rule. For example: `best-practice`, `correctness`,
`maintainability`. For more information, see [Semgrep Registry rule requirements][101].

## `paths`[​][102]

### Exclude a rule in paths[​][103]

To ignore a specific rule on specific files, set the `paths:` key with one or more filters. The
patterns apply to the full file paths relative to the project root.

Example:

`rules:
  - id: eqeq-is-bad
    languages: 
      - python
      - javascript
    severity: MEDIUM
    pattern: $X == $X
    paths:
      exclude:
        - "src/**/*.jinja2"
        - "*_test.go"
        - "project/tests"
        - "project/static/*.js"
`

When invoked with `semgrep -f rule.yaml project/`, the preceding rule runs on files inside
`project/`, but no results are returned for:

* any file with a `.jinja2` file extension
* any file whose name ends in `_test.go`, such as `project/backend/server_test.go`
* any file inside `project/tests` or its subdirectories
* any file matching the `project/static/*.js` glob pattern
note

The glob syntax is from [Python's `wcmatch`][104] and is used to match against the given file and
all its parent directories.

### Limit a rule to paths[​][105]

Conversely, to run a rule *only* on specific files, set a `paths:` key with one or more of these
filters:

`rules:
  - id: eqeq-is-bad
    pattern: $X == $X
    languages: 
      - python
      - javascript
    severity: MEDIUM
    paths:
      include:
        - "*_test.go"
        - "project/server"
        - "project/schemata"
        - "project/static/*.js"
        - "tests/**/*.js"
`

When invoked with `semgrep -f rule.yaml project/`, this rule runs on files inside `project/`, but
results are returned only for:

* files whose name ends in `_test.go`, such as `project/backend/server_test.go`
* files inside `project/server`, `project/schemata`, or their subdirectories
* files matching the `project/static/*.js` glob pattern
* all files with the `.js` extension, arbitrary depth inside the tests folder

If you are writing tests for your rules, add any test file or directory to the included paths as
well.

note

When mixing inclusion and exclusion filters, the exclusion ones take precedence.

Example:

`paths:
  include: "project/schemata"
  exclude: "*_internal.py"
`

The preceding rule returns results from `project/schemata/scan.py` but not from
`project/schemata/scan_internal.py`.

## Additional examples[​][106]

This section contains more complex rules that perform advanced code searching.

### Complete useless comparison[​][107]

`rules:
  - id: eqeq-is-bad
    languages: [python]
    severity: MEDIUM
    patterns:
      - pattern-not-inside: |
          def __eq__(...):
              ...
      - pattern-not-inside: assert(...)
      - pattern-not-inside: assertTrue(...)
      - pattern-not-inside: assertFalse(...)
      - pattern-either:
          - pattern: $X == $X
          - pattern: $X != $X
          - patterns:
              - pattern-inside: |
                  def __init__(...):
                       ...
              - pattern: self.$X == self.$X
      - pattern-not: 1 == 1
    message: "useless comparison operation `$X == $X` or `$X != $X`"
`

The preceding rule makes use of many operators. It utilizes `pattern-either`, `patterns`, `pattern`,
and `pattern-inside` to carefully consider different cases, and employs `pattern-not-inside` and
`pattern-not` to exclude specific unnecessary comparisons.

## Full specification[​][108]

The [full configuration-file format][109] is defined as a [jsonschema][110] object.

Not finding what you need in this doc? Ask questions in our [Community Slack group][111], or see
[Support][112] for other ways to get help.

Tags:

* [Rule writing][113]
[Edit this page][114]
Last updated on Dec 10, 2025

[1]: /docs/writing-rules/overview
[2]: /docs/tags/rule-writing
[3]: https://semgrep.dev/learn
[4]: #schema
[5]: #required
[6]: /docs/contributing/contributing-to-semgrep-rules-repository#rule-messages
[7]: /docs/semgrep-supply-chain/findings#filter-findings
[8]: /docs/writing-rules/rule-syntax#language-extensions-and-languages-key-values
[9]: https://www.pcre.org/current/doc/html/pcre2pattern.html
[10]: #language-extensions-and-languages-key-values
[11]: /docs/semgrep-ce-languages
[12]: /docs/supported-languages#language-maturity-summary
[13]: #optional
[14]: #options
[15]: #fix
[16]: #metadata
[17]: #min-version-and-max-version
[18]: #min-version-and-max-version
[19]: #paths
[20]: #pattern-inside
[21]: #metavariable-regex
[22]: https://docs.python.org/3/library/re.html#re.match
[23]: #metavariable-pattern
[24]: #metavariable-comparison
[25]: https://docs.python.org/3/reference/expressions.html#comparisons
[26]: #metavariable-name
[27]: #pattern-not
[28]: #pattern-not-inside
[29]: #pattern-not-regex
[30]: https://www.pcre.org/current/doc/html/pcre2pattern.html
[31]: #operators
[32]: #pattern
[33]: #patterns
[34]: #patterns-operator-evaluation-strategy
[35]: #pattern-inside
[36]: #pattern
[37]: #pattern-regex
[38]: #pattern-either
[39]: #pattern-not-inside
[40]: #pattern-not
[41]: #pattern-not-regex
[42]: #metavariable-regex
[43]: #metavariable-pattern
[44]: #metavariable-comparison
[45]: #focus-metavariable
[46]: #pattern-either
[47]: https://shattered.io/
[48]: #pattern-regex
[49]: https://www.pcre.org/current/doc/html/pcre2pattern.html
[50]: https://www.pcre.org/current/doc/html/pcre2pattern.html#uniextseq
[51]: #example-pattern-regex-combined-with-other-pattern-operators
[52]: #example-pattern-regex-used-as-a-standalone-top-level-operator
[53]: https://docs.octoprint.org/en/master/configuration/yaml.html#scalars
[54]: https://www.regular-expressions.info/named.html
[55]: #pattern-not-regex
[56]: https://www.pcre.org/current/doc/html/pcre2pattern.html
[57]: #focus-metavariable
[58]: https://semgrep.dev/login
[59]: #including-multiple-focus-metavariables-using-set-intersection-semantics
[60]: /docs/writing-rules/experiments/multiple-focus-metavariables
[61]: #metavariable-regex
[62]: https://www.pcre.org/current/doc/html/pcre2pattern.html
[63]: /docs/writing-rules/pattern-syntax#metavariables
[64]: https://semgrep.dev/playground/s/EbDB
[65]: #metavariable-pattern
[66]: /docs/writing-rules/pattern-syntax#metavariables
[67]: #metavariable-pattern-with-nested-language
[68]: https://semgrep.dev/s/z95k
[69]: https://semgrep.dev/s/pkNk
[70]: #example-match-javascript-code-inside-html
[71]: #example-filter-regex-matches
[72]: #metavariable-comparison
[73]: https://docs.python.org/3/reference/expressions.html#comparisons
[74]: /docs/writing-rules/pattern-syntax#metavariables
[75]: https://semgrep.dev/s/GWv6
[76]: https://docs.python.org/3/reference/expressions.html#boolean-operations
[77]: #legacy-metavariable-comparison-keys
[78]: #metavariable-name
[79]: /docs/getting-started/cli#enable-cross-file-analysis
[80]: #pattern-not
[81]: #pattern-inside
[82]: #pattern-not-inside
[83]: #metavariable-matches
[84]: #metavariables-in-logical-ands
[85]: #metavariables-in-logical-ors
[86]: #metavariables-in-complex-logic
[87]: #options
[88]: /docs/writing-rules/pattern-syntax#associative-and-commutative-operators
[89]: /docs/writing-rules/pattern-syntax#constants
[90]: /docs/writing-rules/data-flow/constant-propagation
[91]: /docs/writing-rules/generic-pattern-matching
[92]: https://en.wikipedia.org/wiki/INI_file
[93]: /docs/writing-rules/generic-pattern-matching
[94]: https://github.com/semgrep/semgrep/blob/develop/interfaces/Rule_options.atd
[95]: #fix
[96]: /docs/writing-rules/autofix
[97]: #metadata
[98]: /docs/contributing/contributing-to-semgrep-rules-repository#fields-required-by-the-security-ca
tegory
[99]: #min-version-and-max-version
[100]: #category
[101]: /docs/contributing/contributing-to-semgrep-rules-repository#semgrep-registry-rule-requirement
s
[102]: #paths
[103]: #exclude-a-rule-in-paths
[104]: https://pypi.org/project/wcmatch/
[105]: #limit-a-rule-to-paths
[106]: #additional-examples
[107]: #complete-useless-comparison
[108]: #full-specification
[109]: https://github.com/semgrep/semgrep-interfaces/blob/main/rule_schema_v1.yaml
[110]: http://json-schema.org/specification.html
[111]: https://go.semgrep.dev/slack
[112]: /docs/support/
[113]: /docs/tags/rule-writing
[114]: https://github.com/semgrep/semgrep-docs/edit/main/docs/writing-rules/rule-syntax.md
