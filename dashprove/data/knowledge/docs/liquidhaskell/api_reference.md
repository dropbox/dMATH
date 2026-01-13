[LiquidHaskell Logo]

LiquidHaskell *(LH)* refines Haskell's types with logical predicates that let you enforce important
properties at compile time.

# Guarantee Functions are Total[¶][1]

LH warns you that head is not total as it is missing the case for `[]` and checks that it is total
on `NonEmpty` lists. [(more...)][2]

The input contract propagates to uses of `head` which are verified by ensuring the arguments are
`NonEmpty`.

# Keep Pointers Within Bounds[¶][3]

LH lets you avoid off-by-one errors that can lead to crashes or buffer overflows. [(more...)][4]

Dependent contracts let you specify, e.g. that `dotProduct` requires equal-sized vectors.

# Avoid Infinite Loops[¶][5]

LH checks that functions terminate and so warns about the infinite recursion due to the missing case
in `fib`. [(more...)][6]

*Metrics* let you check that recursive functions over complex data types terminate.

# Enforce Correctness Properties[¶][7]

Write correctness requirements, for example a list is ordered, as refinements. LH makes illegal
values be *unrepresentable*. [(more...)][8]

LH automatically points out logic bugs, and proves that functions return correct outputs *for all
inputs*.

# Prove Laws by Writing Code[¶][9]

Specify *laws*, e.g. that the append function `++` is associative, as Haskell functions.

Verify laws via *equational proofs* that are plain Haskell functions. Induction is simply recursion,
and case-splitting is just pattern-matching.

# Get Started[¶][10]

The easiest way to try LiquidHaskell is [online, in your browser][11]. This environment is ideal for
quick experiments or following one of the tutorials:

* The [Official Tutorial][12] (long but complete) (has interactive exercises)
* [Andres Loeh's Tutorial][13] (concise but incomplete)

For links to more documentation, see the nav-bar at the top of this page.

# Get Involved[¶][14]

If you are interested in contributing to LH and its ecosystem, that's great! We have more
information on our [GitHub repository][15].

[1]: #guarantee-functions-are-total
[2]: blogposts/2013-01-31-safely-catching-a-list-by-its-tail.lhs
[3]: #keep-pointers-within-bounds
[4]: blogposts/2013-03-04-bounding-vectors.lhs
[5]: #avoid-infinite-loops
[6]: tags.html#termination
[7]: #enforce-correctness-properties
[8]: blogposts/2013-07-29-putting-things-in-order.lhs
[9]: #prove-laws-by-writing-code
[10]: #get-started
[11]: https://liquidhaskell.goto.ucsd.edu/index.html
[12]: https://ucsd-progsys.github.io/liquidhaskell-tutorial/
[13]: https://liquid.kosmikus.org
[14]: #get-involved
[15]: https://github.com/ucsd-progsys/liquidhaskell
