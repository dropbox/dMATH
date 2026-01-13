[The Dafny logo, showing the word Dafny in blue next to wavy black and blue lines.] Dafny is a
verification-aware programming language that has native support for recording specifications and is
equipped with a static program verifier. By blending sophisticated automated reasoning with familiar
programming idioms and tools, Dafny empowers developers to write provably correct code (w.r.t.
specifications). It also compiles Dafny code to familiar development environments such as C#, Java,
JavaScript, Go and Python (with more to come) so Dafny can integrate with your existing workflow.
Dafny makes rigorous verification an integral part of development, thus reducing costly late-stage
bugs that may be missed by testing.

In addition to a verification engine to check implementation against specifications, the Dafny
ecosystem includes several compilers, plugins for common software development IDEs, a LSP-based
Language Server, a code formatter, a reference manual, tutorials, power user tips, books, the
experiences of professors teaching Dafny, and the accumulating expertise of industrial projects
using Dafny.

Dafny has support for common programming concepts such as

* mathematical and bounded integers and reals, bit-vectors, classes, iterators, arrays, tuples,
  generic types, refinement and inheritance,
* [inductive datatypes][1] that can have methods and are suitable for pattern matching,
* [lazily unbounded datatypes][2],
* [subset types][3], such as for bounded integers,
* [lambda expressions][4] and functional programming idioms,
* and [immutable and mutable data structures][5].

Dafny also offers an extensive toolbox for mathematical proofs about software, including

* [bounded and unbounded quantifiers][6],
* [calculational proofs][7] and the ability to use and prove lemmas,
* [pre- and post-conditions, termination conditions, loop invariants, and read/write
  specifications][8].
[A snippet of code shown in the VSCode IDE showing an implementation of the Dutch national flag
problem written in Dafny. IDE extensions are showing successes and failures in verification reported
by the Dafny Language Server running in real time.] Dafny running in Visual Studio Code

[1]: ./latest/DafnyRef/DafnyRef#sec-inductive-datatypes
[2]: ./latest/DafnyRef/DafnyRef#sec-co-inductive-datatypes
[3]: ./latest/DafnyRef/DafnyRef#sec-subset-types
[4]: ./latest/DafnyRef/DafnyRef#sec-lambda-expressions
[5]: ./latest/DafnyRef/DafnyRef#sec-collection-types
[6]: ./latest/DafnyRef/DafnyRef#sec-quantifier-domains
[7]: ./latest/DafnyRef/DafnyRef#sec-calc-statement
[8]: ./latest/DafnyRef/DafnyRef#sec-specification-clauses
