* Z3 JavaScript
On this page

# Z3 JavaScript

The Z3 distribution comes with TypeScript (and therefore JavaScript) bindings
for Z3. In the following we give a few examples of using Z3 through these
bindings. You can run and modify the examples locally in your browser.

info

The bindings do not support all features of z3. For example, you cannot (yet)
create array expressions in the same way that you can create arithmetic
expressions. The JavaScript bindings have the distinct advantage that they let
you use z3 directly in your browser with minimal extra dependencies.

## Warmup[​][1]

We use a collection of basic examples to illustrate the bindings. The first
example is a formula that establishes that there is no number both above 9 and
below 10:

Run

We note that the JavaScript bindings wrap z3 expressions into JavaScript options
that support methods for building new expressions. For example, the method `ge`
is available on an arithmetic expression `a`. It takes one argument `b` and
returns and expression corresponding to the predicate `a >= b`. The `Z3.solve`
method takes a sequence of predicates and checks if there is a solution. If
there is a solution, it returns a model.

## Propositional Logic[​][2]

Prove De Morgan's Law

Run

What not to wear? It is well-known that developers of SAT solvers have
difficulties looking sharp. They like to wear some combination of shirt and tie,
but can't wear both. What should a SAT solver developer wear?

Run

## Integer Arithmetic[​][3]

solve `x > 2 and y < 10 and x + 2y = 7`

Run

### Dogs, cats and mice[​][4]

Given 100 dollars, the puzzle asks if it is possible to buy 100 animals based on
their prices that are 15, 1, and 0.25 dollars, respectively.

Run

## Arrays[​][5]

Arrays use the methods `select` and `store` to access and update elements. Note
that arrays are static and these operations return new arrays.

### Prove `Store(arr, idx, val)[idx] == val`[​][6]

Run

### Find unequal arrays with the same sum[​][7]

We illustrate how to use the solver in finding assignments of array values that
satisfy a given predicate. In this example, we want to find two arrays of length
4 that have the same sum, but are not equal:

Run

## Uninterpreted Functions[​][8]

The method `call` is used to build expressions by applying the function node to
arguments.

### Prove `x = y implies g(x) = g(y)`[​][9]

Run

### Disprove `x = y implies g(g(x)) = g(y)`[​][10]

we illustrate the use of the `Solver` object in the following example. Instead
of calling `Z3.solve` we here create a solver object and add assertions to it.
The `solver.check()` method is used to check satisfiability (we expect the
result to be `sat` for this example). The method `solver.model()` is used to
retrieve a model:

Run

### Prove `x = y implies g(x) = g(y)`[​][11]

Run

### Disprove that `x = y implies g(g(x)) = g(y)`[​][12]

Run

## Solve sudoku[​][13]

The popular Sudoku can be solved.

Run

The encoding used in the following example uses arithmetic. It works here, but
is not the only possible encoding approach. You can also use bit-vectors for the
cells. It is generally better to use bit-vectors for finite domain problems in
z3.

## Arithmetic over Reals[​][14]

You can create constants ranging over reals from strings that use fractions,
decimal notation and from floating point numbers.

Run

Z3 uses arbitrary precision arithmetic, so decimal positions are not truncated
when you use strings to represent real numerals.

Run

## Non-linear arithmetic[​][15]

Z3 uses a decision procedure for non-linear arithmetic over reals. It is based
on Cylindric Algebraic Decomposition. Solutions to non-linear arithmetic
formulas are no longer necessarily rational. They are represented as *algebraic
numbers* in general and can be displayed with any number of decimal position
precision.

Run

## Bit-vectors[​][16]

Unlike in programming languages, there is no distinction between signed and
unsigned bit-vectors. Instead the API supports operations that have different
meaning depending on whether a bit-vector is treated as a signed or unsigned
numeral. These are signed comparison and signed division, remainder operations.

In the following we illustrate the use of signed and unsigned
less-than-or-equal:

Run

It is easy to write formulas that mix bit-wise and arithmetic operations over
bit-vectors.

Run

## Using Z3 objects wrapped in JavaScript[​][17]

The following example illustrates the use of AstVector:

Run
[Edit this page][18]

[1]: #warmup
[2]: #propositional-logic
[3]: #integer-arithmetic
[4]: #dogs-cats-and-mice
[5]: #arrays
[6]: #prove-storearr-idx-validx--val
[7]: #find-unequal-arrays-with-the-same-sum
[8]: #uninterpreted-functions
[9]: #prove-x--y-implies-gx--gy
[10]: #disprove-x--y-implies-ggx--gy
[11]: #prove-x--y-implies-gx--gy-1
[12]: #disprove-that-x--y-implies-ggx--gy
[13]: #solve-sudoku
[14]: #arithmetic-over-reals
[15]: #non-linear-arithmetic
[16]: #bit-vectors
[17]: #using-z3-objects-wrapped-in-javascript
[18]: https://github.com/microsoft/z3guide/tree/main/website/docs-programming/01
 - Z3 JavaScript Examples.md
