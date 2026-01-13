─────────────────────┬──────────────────────────────────┬──────────────────────────
[[Metamath Home]][1] │Metamath Proof Explorer Home Page │[First >][2]              
                     │                                  │[Last >][3]               
─────────────────────┴──────────────────────────────────┴──────────────────────────
[Mirrors][4] > [Home][5] > MPE Home > [Th. List][6] > [Recent][7]                  
───────────────────────────────────────────────────────────────────────────────────
The aleph null above is the symbol for the first infinite cardinal number, discovered by Georg
Cantor in 1873 (see Theorem [aleph0][8]). This is the starting page for the Metamath Proof Explorer
subproject (set.mm database). See the main [Metamath Home Page][9] for an overview of Metamath and
download links. If you wish to contribute your own proofs to the Metamath project, see [How can I
contribute to Metamath?][10]

────────────────────────────────────────────┬───────────────────────────────────────────────────────
Contents of this page                       │Related pages                                          
                                            │                                                       
[Metamath Proof Explorer Overview][11]      │[Theorem List (Table of Contents)][35]                 
                                            │                                                       
[How Metamath Proofs Work][12]              │[Most Recent Proofs (this mirror)][36] ([latest][37])  
                                            │                                                       
[The Axioms][13] ([Propositional            │[Conventions and Style][38]                            
Calculus][14], [Predicate Calculus][15],    │                                                       
[Set Theory][16], [The Tarski–Grothendieck  │[Bibliographic Cross-Reference][39]                    
Axiom][17])                                 │                                                       
                                            │[Definition List (3MB)][40]                            
[The Theory of Classes][18] Added           │                                                       
*13-Dec-2015*                               │[Deduction Form and Natural Deduction][41]             
                                            │                                                       
[A Theorem Sampler][19]                     │[Weak Deduction Theorem][42] (an older method)         
                                            │                                                       
[2 + 2 = 4 Trivia][20] ([more][21])         │[Real and Complex Numbers][43]                         
                                            │                                                       
[Appendix 1: A Note on the Axioms][22]      │[Algebraic and Topological Structures][44] Added       
                                            │*24-Nov-2018*                                          
[Appendix 2: Traditional Textbook Axioms of │                                                       
Predicate Calculus][23]                     │[Frege Notation][45] Added *7-Nov-2020*                
                                            │                                                       
[Appendix 3: Distinct Variables][24]        │[ZFC Axioms With No Distinct Variables][46]            
([History][25], [Notes][26]) Revised        │                                                       
*21-Dec-2016*                               │[ASCII Symbol Equivalents for Text-Only Browsers][47]  
                                            │                                                       
[Appendix 4: A Note on Definitions][27]     │Miscellaneous [notes][48]                              
                                            │                                                       
[Appendix 5: How to Find Out What Axioms a  │[How can I contribute to Metamath?][49]                
Proof Depends On][28]                       │                                                       
                                            │External links                                         
[Appendix 6: Notation for Function and      │                                                       
Operation Values][29]                       │[GitHub repository][50] (contains the database as well 
                                            │as help on contributing)                               
[Appendix 7: Some Predicate Calculus        │                                                       
Subsystems][30]                             │To search this site you can use [Google][51] [retrieved
                                            │21-Dec-2016] restricted to a mirror site. For example, 
[Appendix 8: Axiom Numbering Before December│to find references to infinity enter "infinity         
2018][31]                                   │site:us.metamath.org". More efficient searching is     
                                            │possible with direct use of the [Metamath program][52],
[Reading Suggestions][32]                   │once you get used to its [ASCII tokens][53]. See the   
                                            │wildcard features in "help search" and "help show      
[Bibliography][33]                          │statement".                                            
                                            │                                                       
[Browsers and Fonts][34]                    │                                                       
────────────────────────────────────────────┴───────────────────────────────────────────────────────


Metamath Proof Explorer Overview


*From [this proposition][54] it will follow, when arithmetical addition has been defined, that
1+1=2.*
—*Principia Mathematica*, Volume I, page 360.


────────────────────────────────────────────────────────────────────────────────────────────────────
David A. Wheeler has prepared an excellent 14-minute YouTube video, [Metamath Proof Explorer: A     
Modern Principia Mathematica][55], on the history of formalization and the motivation for Metamath, 
the proof of 2+2=4, and more.                                                                       
────────────────────────────────────────────────────────────────────────────────────────────────────

Inspired by Whitehead and Russell's monumental *Principia Mathematica*, the Metamath Proof Explorer
has over 26,000 completely worked out proofs in its main sections (and over 41,000 counting
"mathboxes", which are annexes where contributors can develop additional topics), starting from the
very foundation that mathematics is built on and eventually arriving at familiar mathematical facts
and beyond. Each proof is pieced together with razor-sharp precision using a simple substitution
rule that practically anyone (with lots of patience) can follow, not just mathematicians. Every step
can be drilled down deeper and deeper into the labyrinth until axioms of logic and set theory—the
starting point for all of mathematics—will ultimately be found at the bottom. You could spend
literally days exploring the astonishing tangle of logic leading, say, from the seemingly mundane
theorem [2+2=4][56] back to these axioms.

Essentially everything that is possible to know in mathematics can be derived from a handful of
axioms known as *Zermelo-Fraenkel set theory,* which is the culmination of many years of effort to
isolate the essential nature of mathematics and is one of the most profound achievements of mankind.

The Metamath Proof Explorer starts with these axioms to build up its proofs. There may be symbols
that are unfamiliar to you, but we show in detail how they are manipulated in the proofs, and in
principle you don't have to know what they mean. In fact, there is a philosophy called *formalism*
which says that mathematics is a game of symbols with no intrinsic meaning. With that in mind,
Metamath lets you watch the game being played and the pieces manipulated according to simple and
precise rules, one step at a time.

As humans, we observe interesting patterns in these "meaningless" symbol strings as they evolve from
the axioms, and we attach meaning to them. One result is the set of natural numbers, whose
properties match those we observe when we count everyday objects, and their extensions to rational
and real numbers. Of course, numbers were discovered centuries before set theory, and historically
they were "reversed engineered" back to the axioms of set theory. The proof of [2 + 2 = 4][57] shows
what was involved in that reverse engineering, representing the work of many mathematicians from
Dedekind to von Neumann. At the other extreme of abstraction is the theory of infinite sets or
transfinite cardinal numbers. Some of the world's most brilliant mathematicians have given us deep
insight into this mysterious and wondrous universe, which is sometimes called "Cantor's paradise."

Metamath's formal proofs are much more detailed than the proofs you see in textbooks. They are
broken down into the most explicit detail possible so that you can see exactly what is going on.
Each proof step represents a microscopic increment towards the final goal. But each step is derived
from previous ones with a very simple rule, and you can verify for yourself the correctness of any
proof with very little skill. All you need is patience. With no prior knowledge of advanced
mathematics or even any mathematics at all, you can jump into the middle of any proof, from the most
elementary to the most advanced, and understand immediately how the symbols were mechanically
manipulated to go from one proof step to another, even if you don't know what the symbols themselves
mean. In the next section we show you how.


How Metamath Proofs Work


*A mathematical theory is not to be considered complete until you have made it so clear that you can
explain it to the first man whom you meet on the street.*
—David Hilbert


────────────────────────────────────────────────────────────────────────────────────────────────────
[The Floating Head of Wisdom says: ] Read this section carefully to learn how to follow a Metamath  
proof.                                                                                              
────────────────────────────────────────────────────────────────────────────────────────────────────

What you need to know The only rule you need to know in order to follow the symbol manipulations in
a Metamath proof is substitution. Substitution consists of replacing the symbols for variables with
expressions representing special cases of those variables. For example, in high-school algebra you
learned that *a* + *b* = *b* + *a*, where *a* and *b* are variables (placeholders for numbers). Two
substitution instances of this law are 5 + 3 = 3 + 5 and (*x* - 7) + *c* = *c* + (*x* - 7). That's
the only mathematical concept you need! Substitution is just writing down a specific example of a
more general formula.

[Note for logicians: The substitution in Metamath proofs is, indeed, simply the direct replacement
of a variable with an expression. The more complex proper substitution of [traditional logic][58] is
a derived concept in Metamath, broken down into multiple primitive steps. [Distinct variable][59]
conditions, which accompany certain axioms and are inherited by theorems, forbid unsound
substitutions.]

How it works To show you how this works in Metamath, we will break down and analyze a proof step in
the proof of 2 + 2 = 4. Once you grasp this example, you will immediately be able to verify for
yourself *any* proof in the database—no further prerequisites are needed. You may not understand
what all (or any) of the symbols mean, but you can follow the rules for how they are manipulated,
like game pieces, to prove theorems.


────────────────────────────────────────────────────────────────────────────────────────────────────
An animated version of the 2+2=4 proof step in this section is presented starting at 7m32s into     
David A. Wheeler's [Metamath Proof Explorer: A Modern Principia Mathematica][60].                   
────────────────────────────────────────────────────────────────────────────────────────────────────

Compare this with the years of study it might take to be able to follow and verify a proof in an
advanced math textbook. Typically such proofs will omit many details, implicitly assuming you have a
deep knowledge of prior material. If you want to be a mathematician, you will still need those years
of study to achieve a high-level understanding. Metamath will not provide you with that. But if you
just want the ability to convince yourself that a string of math symbols that mathematicians call a
"theorem" is a mechanical consequence of the axioms, Metamath's proof method lets you accomplish
that.

Metamath's conceptual simplicity has a tradeoff, which is the often large number of steps needed for
a complete proof all the way back to the axioms. But the proofs have been computer-verified, and you
can choose to study only the steps that interest you and still have complete confidence that the
rest are correct.


────────────────────────────────────────────────────────────────────────────────────────────────────
[Breakdown of a proof step. Credit: N. Megill 2003. Public domain.]                                 
────────────────────────────────────────────────────────────────────────────────────────────────────
Figure 1. Step 2 of the 2p2e4 proof references step 1, which in turn "feeds" the hypothesis of      
earlier theorem oveq2i (which used to be called opreq2i). The conclusion (assertion) of oveq2i then 
generates step 2 of 2p2e4. Carefully note the substitutions (lassoed in thin orange lines) that take
place.                                                                                              
                                                                                                    
*21-Mar-2007* See also Paul Chapman's [Metamath browser screenshot][61], which shows the            
substitutions explicitly.                                                                           
────────────────────────────────────────────────────────────────────────────────────────────────────

In the figure above we show part of the proof of the theorem 2 + 2 = 4, called [2p2e4][62] in the
database. We will show how we arrived at proof step 2, which is an intermediate result stating that
(2 + 2) = (2 + (1 + 1)). (This figure is from an older version of this site that didn't show
indentation levels, and it is less cluttered for the purpose of this tutorial. The indentation
levels and the [little colored numbers][63] can make a higher-level view of the proof easier to
grasp.)

[(1)] Look at Step 2 of the proof. In the Ref column, we see that it references a previously proved
theorem, [oveq2i][64]. The theorem oveq2i requires a hypothesis, and in the Hyp column of Step 2 we
indicate that Step 1 will satisfy (match) this hypothesis.

[(2)] We make substitutions into the variables of the hypothesis of [oveq2i][65] so that it matches
the string of symbols in the Expression column for Step 1. To achieve this, we substitute the
expression "2" for variable 𝐴 and the expression "(1 + 1)" for variable 𝐵. The middle symbol in the
hypothesis of [oveq2i][66] is "=", which is a constant, and we are not allowed to substitute
anything for a constant. Constants must match exactly.

Variables are always colored, and constants are always black (except the gray turnstile ⊢ , which
you may ignore). This makes them easy to recognize. In our example, the purple uppercase italic
letters are variables, whereas the symbols "(", ")", "1", "2", "=", and "+" are constants.

In this example, the constants are probably familiar symbols. In other cases they may not be. You
should focus only on whether the symbols are variables or constants, not on what they "mean." Your
only goal is to determine what substitutions into the variables of the referenced theorem are needed
to make the symbol strings match.

[(3)] In the Expression column of the Assertion box of [oveq2i][67], there are four variables, 𝐴, 𝐵,
𝐶, and 𝐹. Because we have already made substitutions into the hypothesis, variables 𝐴 and 𝐵 have
been committed to the assignments "2" and "(1 + 1)" respectively, and we can't change these
assignments. However, the new variables 𝐶 and 𝐹 are free to be assigned with any expression we want
(subject to the legal syntax requirement described below). By substituting "2" for 𝐶 and "+" for 𝐹,
we end up with (2 + 2) = (2 + (1 + 1)) that we show in the Expression column for Step 2 of the proof
of [2p2e4][68].

[It may seem peculiar to substitute a + sign for a variable, because you wouldn't do that in
high-school algebra. We can do this because the variables represent arbitrary objects called
"classes," not just numbers. See the description for operation value [df-ov][69] (don't worry about
right-hand side of the definition, for now). A number and a + sign are both classes. You have to
free your mind to forget about high-school algebra—pretend you have no idea what a number or "+"
is—and just look at what happens to the symbols, independent of any meaning. In fact (and
ironically), it may be better to look at a proof where all the symbols are unfamiliar, perhaps
[aleph1re][70], so that you can observe the mechanical symbol substitutions without the distraction
of preconceived notions.]

When we first created the proof, why did we choose these particular substitutions for 𝐶 and 𝐹? The
reason is simple—they make the proof work! But how did we know these particular substitutions should
be picked, and not others? That's the hard part—we didn't know, nor did we know that oveq2i should
be referenced in the second proof step, nor did we know that Step 1 would have the right expression
to match the hypothesis of oveq2i. The choices were made using intelligent guesses, that were then
verified to work. This is a skill a mathematician develops with experience. Some of the proofs in
our database were discovered by famous mathematicians. The Metamath Proof Explorer recaptures their
efforts and shows you in complete detail the proof steps and substitutions already worked out. This
allows you to follow a proof even if you are not a mathematician, and be convinced that its
conclusion is a consequence of the axioms.

Sometimes a referenced theorem (or axiom or definition) has no hypotheses. In that case we omit
[(1)] and [(2)] above and immediately proceed to [(3)]. When there are multiple hypotheses, we
repeat [(1)] and [(2)] for each one.


────────────────────────────────────────────────────────────────────────────────────────────────────
[The Floating Head of Wisdom says: ] Done! You should now be able to figure out any Metamath proof. 
In other words, you should be able to draw a diagram like the one above for any proof step of any   
proof.                                                                                              
────────────────────────────────────────────────────────────────────────────────────────────────────

You may want to practice the above procedure for a few other proof steps to make sure you have
grasped the idea.

The rest of this section has some notes on substitutions that you may find helpful and describes the
additional requirements for correctness not mentioned above. As you will observe if you study a few
proofs, the Metamath proof verifier has already ensured these requirements are met, so ordinarily
you don't have to worry about them.

Notes on substitutions

Substitutions are simultaneous. In other words each occurrence of a given variable in a referenced
theorem must be replaced with the same expression. For example, there are two occurrences of 𝐹 in
the Assertion of oveq2i, and both occurrences must be replaced with the same expression, which is
"+" in the above example.

Substitutions are made into the variables of the referenced theorem only, never into the variables
of any proof step referenced in the Hyp column (of the theorem being proved). *In other words you
should pretend that all variables in the theorem being proved are constants for the purpose of
figuring out the substitutions.* You can see this by looking at examples such as Theorem
[idALT][71]. To follow the proof of [idALT][72], you should treat the symbol 𝜑 as if it were a
constant symbol, when you are figuring out the substitutions to make into the variables of the
referenced theorems (or axioms).

If the variables of a referenced theorem (or axiom) happen to have the same names as those in the
theorem being proved, you may want to temporarily rename the variables in the referenced theorem (or
axiom) before substituting expressions for them, to avoid confusion. For example, the proof of
[idALT][73] will be less confusing if the occurrences of 𝜑 in the referenced axioms are renamed to
something else. Specifically, you can rewrite [ax-1][74] as, say, (𝜒 → (𝜓 → 𝜒)). Then, to obtain
step 2 of the proof of [idALT][75], substitute "𝜑" for 𝜒 and "(𝜑 → 𝜑)" for 𝜓.

Legal syntax There is a further requirement for Metamath substitutions we have not described yet.
You can't substitute just any old string of symbols for a purple class variable. Instead, the symbol
string must qualify as a class expression. For example, it would be illegal to substitute the
nonsensical "(1 +" for variable 𝐵 above. However, "(1 + 1)" is legal. Here is how you can tell. "1"
is a legal class by [c1][76]. "+" is a legal class by [caddc][77]. Then, by making these class
substitutions into the class variables of [co][78], we see that "(1 + 1)" is a legal class. But
there is no such construction that would let us show that the nonsensical "(1 +" is a legal class.

Similarly, blue wff variables and red setvar variables can be substituted only with expressions that
qualify as those types.

In other words, we must "prove" that any expression we want to substitute for a variable qualifies
as a legal expression for that type of variable, before we are allowed to make the substitution.
This also states precisely what is being substituted, preventing any ambiguity.

The actual proofs stored in the database have additional steps that construct, from syntax
definitions, the expressions that are substituted for variables. We suppress these construction
steps on the web pages because they would make the proofs very long and tedious. However, the syntax
breakdown is straightforward to check by hand if you make use of the "Syntax hints" provided with
each proof. Once you get used to the syntax, you should be able to "see" its breakdown in your head;
in the meantime you can trust that the Metamath proof verifier did its job.

If you want to see for yourself the hidden steps that construct the variable substitutions for each
proof step, you can display them using the Metamath program. For the proof above, use the commands
"save proof 2p2e4 /normal" followed by "show proof 2p2e4 /all" in the Metamath program. (Follow the
instructions for downloading and running the [Metamath program][79]. Try it, it's easy!) In the
"/all" proof display, you will see that step 21 corresponds to step 2 of the figure above. Steps
14-17 are the hidden steps showing that "(1 + 1)" is a legal class as we described above. To see the
substitutions we talked about for step 2, you can type "show proof 2p2e4 /detailed_step 21".

This is a general property of Metamath, not just of the set.mm database. For example, step 32 of
proof th1 in demo database demo0.mm must match two expressions to use theorem mp:

* ⊢ (𝑃 → 𝑄)
* ⊢ ((𝑡 + 0) = 𝑡 → ((𝑡 + 0) = 𝑡 → 𝑡 = 𝑡))
This could be matched many different expressions, for example:

* 𝑃:= (𝑡 + 0) = 𝑡
* 𝑄:= ((𝑡 + 0) = 𝑡 → 𝑡 = 𝑡)
or

* 𝑃:= (𝑡 + 0) = 𝑡 → ((𝑡 + 0) = 𝑡
* 𝑄:= 𝑡 = 𝑡)

However, theorem mp in database demo0.mm actually has four arguments (which you can see if you ask
metamath.exe to show them). The first two are the substitutions, which are usually suppressed. That
is, you first have to prove wff 𝑃, then wff 𝑄, then ⊢ 𝑃, then ⊢ (𝑃 → 𝑄), and then theorem mp applies
to derive ⊢ 𝑄. If you use metamath.exe, "show proof th1" will list only 5 steps, but with suspicious
gaps in the numbering; "show proof th1 /all" will show all the substitution steps (which in fact
make up the majority of the actual proof on disk).

A proof of wff 𝑃 is equivalent to a proof that 𝑃 is a well-formed formula, and it is this that
prevents invalid substitutions like 𝑄:= 𝑡 = 𝑡); you will not be able to prove wff 𝑡 = 𝑡) so that
proof will not work. But in any case the Metamath verifier isn't being asked to come up with the
substitution (although most metamath proof assistants will, using unification), so no matching
algorithm is necessary, only a substitution and an equality check.

In the case of axioms and definitions, we *do* show their detailed syntax breakdown, because there
is free space on those web pages not used for anything else. These can help you become familiar with
the syntax. For example, look at the set.mm (Metamath Proof Explorer) definition of the number 2,
[df-2][80]. You can see, at Step 4, the demonstration that (1 + 1) is a legal expression that
qualifies as a class, i.e. that can be substituted for a purple class variable.

Distinct variable conditions Our final requirement for substitutions is described in [Appendix 3:
Distinct Variables][81] below. These restrictions have no effect on how you figure out the the
substitutions that were made in a proof step. All they do is prohibit certain substitutions that
would otherwise be legal based what we have described so far. Eventually you should learn how they
work in order to complete your understanding of the mechanics of logic, but for now, you can trust
that the Metamath proof verifier has ensured that they have been met.

Class variables Our example of 2+2=4, with its purple class variables, depends on a definitional
mechanism that extends the wff and setvar variables used in the axioms to greatly simplify our
presentation. After the axiom section below, we describe the [theory of classes][82], which you
should read to understand how these tie into the primitive concepts used by the axioms.


The Axioms


*Perfection is when there is no longer anything more to take away.*
—Antoine de Saint-Exupery

[The material in this section is intended to be self-contained. However, you may also find it
helpful to review [these suggestions][83]. A more extensive but still informal overview is given in
Chapter 3, "Abstract Mathematics Revealed," of the [ *Metamath* book][84] (1.3 MB PDF file; click on
the fourth bookmark in your PDF reader).]

An axiom is a fundamental assumption that provides a starting point for reasoning. The axioms for
(essentially) all of mathematics can be conveniently divided into three groups: propositional
calculus, predicate calculus, and set theory. Each axiom is a string of mathematical symbols of two
kinds: constants, also called connectives, which we show in black; and variables, which we show in
color. The constants that occur in the axioms of predicate calculus are (, ), →, ¬, =, ∈, and ∀
(left parenthesis, right parenthesis, implies, not, equals, is a member of, for all). The axioms of
set theory may also use some defined constants (like ∧ and ∃) when this results in a more readable
assertion.

Variables are placeholders that can be substituted with other expressions (strings of symbols).
There are two kinds of variables in the axioms, setvar variables (lowercase italic letters in red)
and wff ("well-formed formula") variables (lowercase Greek letters in blue). A wff variable can be
substituted with any expression qualifying as a wff (see below). A setvar variable can be
substituted only with another setvar variable (in other words with an expression of length one,
whose only symbol is a setvar variable) since there are no rules that let us construct more complex
expressions for them. We call these "setvar variables" rather than just "set variables" to emphasize
this restriction. The reason for this restriction is that they often serve as bound or dummy
variables in expressions, i.e. the first argument of the ∀ quantifier connective introduced below.
It is the same reason that you cannot substitute say 2 for x in "the integral of x dx" to result in
"the integral of 2 d2", which would be nonsense.

[In later proofs you will see a third kind of variable, called a class variable, which is shown in
purple (usually as uppercase italic letters) and is a kind of generalization of the setvar variable
that can be substituted with expressions such as 2 or (π + 3). The [theory of classes][85] will be
discussed in the next section.]

Following the tradition in the literature, we use italic Greek letters for wff variables and
lower-case italic letters for setvar variables.

[If you use a monochrome display or printout, or are colorblind, the red variables are lowercase
italic letters, the purple ones uppercase italic, and the blue ones italic Greek. Sometimes we use
mathematical symbols (such as +) as class variables, and in that case they are distinguished by both
the purple color and a dotted underline. In all cases, the colors are not necessary to disambiguate
the symbols. You can see the list of all symbols including variables on the [ASCII Symbol
Equivalents][86] page.]

Any mathematical object whatsoever, such as the number 2, the operation of taking a square root, or
the surface of a sphere, is considered to be a set in set theory. The red setvar variables represent
arbitrary sets such as these. (You can't substitute the expressions for these sets directly into the
setvars, as discussed above, but must use the setvars as part of a formula that specifies the
properties of these sets indirectly. Our theory of classes in the next section makes direct
substitutions possible, giving us an often more convenient mechanism for practical work.)

*A set can be equal to another set, can be contained in another set, and can contain other sets.*
For example, the set of real numbers contains, of course, all of the real numbers. A specific real
number such as 2 is also a set, but not in such a familiar way—it contains a very complex
construction of sets, a kind of machine inside of it that causes it to behave according to the laws
for real numbers in the presence of the ZFC axioms. The square root operation is a set containing an
infinite number of ordered pairs, one for each nonnegative real number; the first member of each
pair is the number and the second member its square root. (Recall that a setvar variable can only be
replaced with another setvar variable, so we cannot replace a setvar variable with say the symbol
"2". Manipulation of such symbols uses the definitional mechanism we will introduce in the [Theory
of Classes][87] section below.)

A wff is an expression (string of symbols) constructed as follows. A starting wff either is a wff
variable, or it consists of two setvar variables connected with either = ("equals," "is identical
to") or ∈ ("is an element of," "is a member of," "belongs to," "is contained in"). Two wffs may be
connected with → ("implies," "only if") to form a new wff, with parentheses around the result to
prevent ambiguity. A wff may be prefixed with ¬ ("not") to form a new wff. And finally, a wff may be
prefixed with ∀ ("for all," "for each," "for every") and a setvar variable to form a new wff. *To
summarize:* If 𝜑 is a wff variable and 𝑥 and 𝑦 are setvar variables, then 𝜑, 𝑥 = 𝑦, and 𝑥 ∈ 𝑦 are
(starting) wffs. If 𝜑 and 𝜓 are wffs and 𝑥 is a setvar variable, then (𝜑 → 𝜓), ¬ 𝜑, and ∀𝑥𝜑 are also
wffs.

Note that a wff variable can be viewed as a wff in its own right as well as a placeholder for which
other wffs can be substituted.

The axioms below are examples of wffs. Each page describing an axiom, for example [ax-12][88],
presents the axiom's construction in a syntax breakdown chart, showing that it follows these rules
and is therefore a wff. You may want to look at a few of these to make sure you understand how wffs
are constructed and how to deconstruct, or parse, them into their components.

*Wffs are either true or false,* depending on what is assigned to the variables they contain. For
example, the wff 𝑥 = 𝑦 is true if 𝑥 and 𝑦 stand for the same set and false otherwise—there is no
in-between.

An axiom (example: [ax-1][89]) is a wff that we postulate to be true no matter what (within the
constraints of the syntax rules) we substitute for its variables. An inference rule (example:
[ax-mp][90]) consists of one or more wffs called hypotheses, together with a wff called the
conclusion (which on our web pages with proofs we also call its assertion) that we postulate to be
true provided the hypotheses are true. A proof is a sequence of substitution instances of axioms and
inference rules, where the hypotheses of the inference rules match previous steps in the sequence.
The last step in a proof is a theorem (example: [idALT][91]). For brevity, a proof may also refer to
earlier theorems (example: [id][92]), but in principle it can be expanded into references to only
the initial axioms and rules.

We also use the word "theorem" informally to denote the result of a proof that also allows
references to local hypotheses and thus has the form of an inference rule (example: [a1i][93]);
however, strictly speaking, such a theorem should be called a derived inference rule or deduction.
In a derived inference rule, a proof is a sequence steps each of which is a substitution instance of
an axiom, a hypothesis, or a substitution instance of an inference rule applied to previous steps.
(Note that a proof with nothing but references to hypotheses still qualifies as a proof, even though
neither axioms nor inference rules are involved. For example, the unusual derived inference rule
[a1ii][94] is proved before we even introduce the first axiom!)

Propositional calculus deals with general truths about wffs regardless of how they are constructed.
The simplest propositional truth is "(𝜑 → 𝜑)", which can be read "if something is true, then it is
true"—rather trivial and obvious, but nonetheless it must be proved from the axioms (see theorem
[id][95]). In an application of this truth, we could replace 𝜑 with a more specific wff to give us,
for example, "(𝑥 = 𝑦 → 𝑥 = 𝑦)". (You might think that [id][96] would be a useless bit of trivia, but
in fact it is frequently used. For example, look at its use in the proof of the law of assertion
[pm2.27][97].)

Our system of propositional calculus consists of three axioms and the modus ponens inference rule.
It is attributed to Jan Łukasiewicz (pronounced *woo-kah-SHAY-vitch*) and was popularized by Alonzo
Church, who called it system P2. *(Thanks to Ted Ulrich for this information.)*


─────────────────────────┬──────┬─────────────────────────────────────
[ Axiom *Simp*][98]      │ax-1  │⊢ (𝜑 → (𝜓 → 𝜑))                      
─────────────────────────┼──────┼─────────────────────────────────────
[Axiom *Frege*][99]      │ax-2  │⊢ ((𝜑 → (𝜓 → 𝜒)) → ((𝜑 → 𝜓) → (𝜑 →   
                         │      │𝜒)))                                 
─────────────────────────┼──────┼─────────────────────────────────────
[Axiom *Transp*][100]    │ax-3  │⊢ ((¬ 𝜑 → ¬ 𝜓) → (𝜓 → 𝜑))            
─────────────────────────┼──────┼─────────────────────────────────────
[Rule of Modus           │ax-mp │⊢ 𝜑 & ⊢ (𝜑 → 𝜓) ⇒ ⊢ 𝜓                
Ponens][101]             │      │                                     
─────────────────────────┴──────┴─────────────────────────────────────

The turnstile ⊢ is a symbol (introduced by Frege in 1879) used in formal logic to indicate that the
wff that follows is provable (and is traditionally used even for an axiom, which is "provable" in
one step, itself); it can be ignored for informal reading. (Technically, the Metamath program needs
an initial token to disambiguate the kind of expression that follows. I figured, why not make it the
turnstile that logic books use? In the Quantum Logic Explorer, on the other hand, I mapped it to a
blank image because my physics colleague didn't like it.) The symbols & and ⇒ are used informally in
the table above to indicate the relationship between hypotheses and conclusion; they are not part of
the formal language.

Predicate calculus introduces three new symbols, = ("equals"), ∈ ("is a member of"), and ∀ ("for
all"). The first two are called "predicates." A predicate specifies a true or false relationship
between its two arguments. As an example, 𝑥 = 𝑥 always evaluates to true regardless of what 𝑥 is, as
Theorem [equid][102] demonstrates. The "for all" symbol, also called the "universal quantifier,"
ranges over or "scans" all possible values of its first (setvar variable) argument, applying them to
the second (wff expression) argument, and returns true if and only if the second argument is true
for every value of the first. The wff ∀𝑥𝜑 is read "for all x, it is the case that phi is true." The
axioms of predicate calculus express the logical properties of these new symbols that are
universally true, independent from any theory (like set theory) making use of them. (What we call
"setvar variables" are more generally called "individual variables" in predicate calculus without
set theory.)

Our axiom system is closely related to a system of Tarski (see the [note on the axioms][103] below).
Our system is exactly equivalent to the [traditional][104] axiom system in most logic textbooks but
has the advantage of being easy to manipulate with a computer program. Each of our axioms
corresponds to an [axiom scheme][105] of the traditional system.

─────────────────────────────────────────┬──────┬───────────────────────────────────────────────────
[Rule of Generalization][106]            │ax-gen│⊢ 𝜑 ⇒ ⊢ ∀𝑥𝜑                                        
─────────────────────────────────────────┼──────┼───────────────────────────────────────────────────
[Quantified Implication][107]            │ax-4  │⊢ (∀𝑥(𝜑 → 𝜓) → (∀𝑥𝜑 → ∀𝑥𝜓))                        
─────────────────────────────────────────┼──────┼───────────────────────────────────────────────────
[Distinctness][108]                      │ax-5  │⊢ (𝜑 → ∀𝑥𝜑), where 𝑥 does not occur in             
                                         │      │𝜑                                                  
─────────────────────────────────────────┼──────┼───────────────────────────────────────────────────
[Existence][109]                         │ax-6  │⊢ ¬ ∀𝑥¬ 𝑥 = 𝑦                                      
─────────────────────────────────────────┼──────┼───────────────────────────────────────────────────
[Equality][110]                          │ax-7  │⊢ (𝑥 = 𝑦 → (𝑥 = 𝑧 → 𝑦 = 𝑧))                        
─────────────────────────────────────────┼──────┼───────────────────────────────────────────────────
[Left Equality for Binary Predicate][111]│ax-8  │⊢ (𝑥 = 𝑦 → (𝑥 ∈ 𝑧 → 𝑦 ∈ 𝑧))                        
─────────────────────────────────────────┼──────┼───────────────────────────────────────────────────
[Right Equality for Binary               │ax-9  │⊢ (𝑥 = 𝑦 → (𝑧 ∈ 𝑥 → 𝑧 ∈ 𝑦))                        
Predicate][112]                          │      │                                                   
─────────────────────────────────────────┴──────┴───────────────────────────────────────────────────
                                                                                                    
                                                                                                    
───────────────────────────┬─────┬──────────────────────────────────────────────────────────────────
[Quantified Negation][113] │ax-10│⊢ (¬ ∀𝑥𝜑 → ∀𝑥¬ ∀𝑥𝜑)                                               
───────────────────────────┼─────┼──────────────────────────────────────────────────────────────────
[Quantifier                │ax-11│⊢ (∀𝑥∀𝑦𝜑 → ∀𝑦∀𝑥𝜑)                                                 
Commutation][114]          │     │                                                                  
───────────────────────────┼─────┼──────────────────────────────────────────────────────────────────
[Substitution][115]        │ax-12│⊢ (𝑥 = 𝑦 → (∀𝑦𝜑 → ∀𝑥(𝑥 = 𝑦 →                                      
                           │     │𝜑)))                                                              
───────────────────────────┼─────┼──────────────────────────────────────────────────────────────────
[Quantified Equality][116] │ax-13│⊢ (¬ 𝑥 = 𝑦 → (𝑦 = 𝑧 → ∀𝑥 𝑦 =                                      
                           │     │𝑧))                                                               
───────────────────────────┴─────┴──────────────────────────────────────────────────────────────────

Axiom [ax-5][117] has a restriction on what substitutions we are allowed to make into its variables.
This restriction is inherited by theorems that use this axiom, according to the substitutions made
in their proofs, and you will often see [distinct variable][118] conditions associated with such
theorems.

The above table is divided into two groups, those from a system devised by logician Alfred Tarski
and some additional auxilliary axiom schemes. The latter, while logically redundant at the object
language level, endow the the system with a property called "scheme completeness", which means that
any theorem (scheme) expressible in the [wff syntax][119] described above can be proved *directly*,
without having to resort to techniques like combining cases or induction on formula length. This is
explained [below][120].

In December 2018, the axioms were renumbered and may not correspond to the numbers in older papers,
slide presentations, and websites. See [Appendix 8][121] for details.

Some definitions will simplify our presentation of the set theory axioms, although in principle they
could be eliminated. Specifically,

* (𝜑 ∧ 𝜓) is an abbreviation for ¬ (𝜑 → ¬ 𝜓)
* (𝜑 ↔ 𝜓) is an abbreviation for ((𝜑 → 𝜓) ∧ (𝜓 → 𝜑))
* ∃𝑥𝜑 is an abbreviation for ¬ ∀𝑥 ¬ 𝜑
In our database, these are introduced (in a different order) by the statements [df-bi][122]
(biconditional), [df-an][123] (logical AND), and [df-ex][124] (existential quantifier). To
understand how definitions are introduced and justified in Metamath, see the [note on
definitions][125] below. But for our purposes, you can just *assume that* [df-bi][126],
[df-an][127], *and* [df-ex][128] *are additional axioms* (which they are in some axiomatizations of
logic) so that you don't have to be concerned with their metalogical justification. Under this
assumption, our primitive or basic connectives are extended with ↔, ∧, and ∃, and our wff
construction rules are extended with [wb][129], [wa][130], and [wex][131]. Alternately, you can
rewrite the set theory axioms below with these three definitions eliminated.

Set theory uses the formalism of propositional and predicate calculus to assert properties of
arbitrary mathematical objects called "sets." A set can be an element of another set, and this
relationship is indicated by the ∈ symbol. Starting with the simplest mathematical object, called
the empty set, set theory builds up more and more complex structures whose existence follows from
the axioms, eventually resulting in extremely complicated sets that we identify with the real
numbers and other familiar mathematical objects. These axioms were developed in response to
[Russell's Paradox][132], a discovery that revolutionized the foundations of mathematics and logic.

Except for Extensionality, the axioms basically say, "given an arbitrary set 𝑥 (and, in the cases of
Replacement and Regularity, provided that an antecedent is satisfied), there exists another set 𝑦
based on or constructed from it, with the stated properties." (The axiom of extensionality can also
be restated this way as shown by [axexte][133].) The individual axiom links provide more detailed
descriptions. We derive the redundant ZF axioms of [Separation][134], [Null Set][135], and
[Pairing][136] from the others as theorems.

In the ZFC axioms that follow, *all setvar variables are assumed to be* [distinct][137]. If you
click on their links you will see the explicit distinct variable conditions. (There is also an
[unusual formalization][138] of these axioms that does not require that their variables be
distinct.) There are no restrictions on the variables that may occur in wff 𝜑 below.


─────────────────────────────────┬─────┬────────────────────────────────────────────────────────────
[Axiom of Extensionality][139]   │ax-ex│⊢ (∀𝑧(𝑧 ∈ 𝑥 ↔ 𝑧 ∈ 𝑦) → 𝑥 = 𝑦)                               
                                 │t    │                                                            
─────────────────────────────────┼─────┼────────────────────────────────────────────────────────────
[Axiom of Replacement][140]      │ax-re│⊢ (∀𝑤∃𝑦∀𝑧(∀𝑦𝜑 → 𝑧 = 𝑦) → ∃𝑦∀𝑧(𝑧 ∈ 𝑦 ↔ ∃𝑤(𝑤 ∈ 𝑥 ∧ ∀𝑦𝜑)))     
                                 │p    │                                                            
─────────────────────────────────┼─────┼────────────────────────────────────────────────────────────
[Axiom of Power Sets][141]       │ax-po│⊢ ∃𝑦∀𝑧(∀𝑤(𝑤 ∈ 𝑧 → 𝑤 ∈ 𝑥) → 𝑧 ∈ 𝑦)                           
                                 │w    │                                                            
─────────────────────────────────┼─────┼────────────────────────────────────────────────────────────
[Axiom of Union][142]            │ax-un│⊢ ∃𝑦∀𝑧(∃𝑤(𝑧 ∈ 𝑤 ∧ 𝑤 ∈ 𝑥) → 𝑧 ∈ 𝑦)                           
─────────────────────────────────┼─────┼────────────────────────────────────────────────────────────
[Axiom of Regularity             │ax-re│⊢ (∃𝑦𝑦 ∈ 𝑥 → ∃𝑦(𝑦 ∈ 𝑥 ∧ ∀𝑧(𝑧 ∈ 𝑦 → ¬ 𝑧 ∈ 𝑥)))               
(Foundation)][143]               │g    │                                                            
─────────────────────────────────┼─────┼────────────────────────────────────────────────────────────
[Axiom of Infinity][144]         │ax-in│⊢ ∃𝑦(𝑥 ∈ 𝑦 ∧ ∀𝑧(𝑧 ∈ 𝑦 → ∃𝑤(𝑧 ∈ 𝑤 ∧ 𝑤 ∈ 𝑦)))                 
                                 │f    │                                                            
─────────────────────────────────┼─────┼────────────────────────────────────────────────────────────
[Axiom of Choice][145]           │ax-ac│⊢ ∃𝑦∀𝑧∀𝑤((𝑧 ∈ 𝑤 ∧ 𝑤 ∈ 𝑥) → ∃𝑣∀𝑢(∃𝑡((𝑢 ∈ 𝑤 ∧ 𝑤 ∈ 𝑡) ∧ (𝑢 ∈ 𝑡 
                                 │     │∧ 𝑡 ∈ 𝑦)) ↔ 𝑢 = 𝑣))                                         
─────────────────────────────────┴─────┴────────────────────────────────────────────────────────────

The ZFC axioms have several equivalent formalizations, and we generally selected ones that are the
shortest to state in terms of set theory primitives. Our [ax-inf][146] and [ax-ac][147] are slighly
unconventional in order to make them shorter. We also provide [ax-inf2][148] and [ax-ac2][149] for
people who want to work with equivalents to the conventional versions. Specifically, [ax-reg][150]
is needed to derive [ax-inf2][151] from [ax-inf][152] and [ax-ac][153] from [ax-ac2][154]. The
reverse derivations do not need [ax-reg][155].

There you have it, the axioms for (essentially) all of mathematics! Marvel at them and stare at them
in awe. If you keep a copy in your wallet, you will carry with you the encoding for all theorems
ever proved and that ever will be proved, from the most mundane to the most profound.

*Note.* Books sometimes make statements like "(essentially) all of mathematics can be derived from
the ZFC axioms." This should not be taken literally—there's not much you can do with these 7 axioms
by themselves! The authors are assuming that you will combine the ZFC axioms with logic (that is,
the axioms and rules of propositional and predicate calculus). Logic and ZFC comprise a total of 20
axioms and 2 rules in our system.

The Tarski–Grothendieck Axiom Above we qualified the phrase "all of mathematics" with "essentially."
The main important missing piece is the ability to do category theory, which requires huge sets
(inaccessible cardinals) larger than those postulated by the ZFC axioms. The Tarski–Grothendieck
Axiom postulates the existence of such sets. We have included it in a separate table below for two
reasons: first, it is not normally considered to be part of ZFC set theory, and second, unlike the
ZFC axioms, it is not "elementary," in that the known forms of it are very long when expanded to set
theory primitives. Below we show it compacted with definitions. Theorem [grothprim][156] shows you
what it looks like when the definitions are expanded into the primitives used by the other axioms.
The Tarski-Grothendieck axiom can be viewed as a very strong replacement of the Axiom of Infinity
and implies that axiom (Theorem [grothomex][157]) as well as the Axiom of Choice (Theorem
[grothac][158]) and the Axiom of Power Sets (Theorem [grothpw][159]).


───────────────────────────┬──────┬─────────────────────────────────────────────────────────────────
[The Tarski–Grothendieck   │ax-gro│⊢ ∃𝑦(𝑥 ∈ 𝑦 ∧ ∀𝑧 ∈ 𝑦(∀𝑤(𝑤 ⊆ 𝑧 → 𝑤 ∈ 𝑦) ∧ ∃𝑤 ∈ 𝑦∀𝑣(𝑣 ⊆ 𝑧 → 𝑣 ∈ 𝑤)) 
Axiom][160]                │th    │∧ ∀𝑧(𝑧 ⊆ 𝑦 → (𝑧 ≈ 𝑦 ∨ 𝑧 ∈ 𝑦)))                                   
───────────────────────────┴──────┴─────────────────────────────────────────────────────────────────

What else is missing from our axioms? Gödel showed that no finite set of axioms or axiom schemes can
completely describe any consistent theory strong enough to include arithmetic. But practically
speaking, the ones above are the accepted foundation that almost all mathematicians explicitly or
implicitly base their work on.

Set theorists often study the consequences of additional axioms that assert, for example, the
existence of larger and larger infinities beyond those postulated by ZFC or even the
Tarski–Grothendieck Axiom, to the point of flirting with inconsistency (although Gödel also showed
that we can never know whether even the ZFC axioms are consistent). While this work is certainly
profound and important, these axioms are not ordinarily used outside of set theory. To study such an
additional axiom with Metamath, we can assert it as a new axiom, or alternately we can simply state
it as a hypothesis to any theorem making use of it.


The Theory of Classes In [Figure 1][161] above that shows part of the proof of 2+2=4, there are some
purple-colored variables such as 𝐴 and 𝐵 that do not appear in any of the axioms of logic and set
theory that we just presented. Even the number 2 is a symbol (a constant) that we cannot manipulate
with the axioms, because of the rule that only setvar variables (and not constants or other
expressions) can be substituted for setvar variables. You may at this point feel lost, wondering how
our introductory 2+2=4 example bears any relationship at all to the axioms! In this section, we will
discuss how this notation ties into our axioms and how to convert it to the primitive or basic
language used by the axioms.

In principle, mathematics can be done using only the setvar and wff variables that appear in the
axioms. But the notation can be awkward to work with and is not always intuitive for humans. To make
mathematics more practical, we extend the set theory language with a definitional notation called
the theory of classes. An expression of the form {𝑥 ∣ 𝜑}, where 𝑥 is any setvar variable and 𝜑 is
any wff, is interchangeably called a class builder, class abstract[ion], class term, or class
comprehension by different authors. We call an object that can be represented in this form a class
and read {𝑥 ∣ 𝜑} as "the class of all 𝑥 such that 𝜑 holds". Every class is a (possibly empty)
collection of sets, although not every such collection is itself a set (see the discussion for
Theorem [ru][162]). For example, {𝑥 ∣ (𝑥 ∈ ℤ ∧ 2 < 𝑥)} is the class (and also set) of all integers
greater than 2.

We use upper-case variables 𝐴, 𝐵,... to range over class builders and call them class variables.
More generally, we let them range over any object representing a class, such as other class
variables. (A class variable 𝐴 itself can be represented with the class builder {𝑥 ∣ 𝑥 ∈ 𝐴} and thus
qualifies as a class; see Theorem [abid2][163].) Our language will now have three kinds of
variables: 𝜑, 𝜓,... ranging over wffs, 𝑥, 𝑦,... ranging over setvars, and 𝐴, 𝐵,... ranging over
classes.

In the basic language, the ∈ and = connectives must connect two setvar variables, so all starting
wffs (without wff variables) are of the form 𝑥 ∈ 𝑦 and 𝑥 = 𝑦. We extend the language syntax, i.e.
the definition of a wff, by allowing ∈ and = to connect two expressions representing classes, as
exemplified in the three definitions below. Theorems involving the extended syntax are derived
making use of those three definitions.

We can construct new classes from existing ones, a property we often exploit to define new concepts.
For example, we introduce the symbol ∪ and define the union of two classes 𝐴 and 𝐵 in [df-un][164]:
(𝐴 ∪ 𝐵) = {𝑥 ∣ (𝑥 ∈ 𝐴 ∨ 𝑥 ∈ 𝐵)}. Here the defined expression on the left of the "=" abbreviates the
expression on the right. When eliminating the defined expression, we can choose for 𝑥 any variable
not occurring in 𝐴 or 𝐵 (Theorem [unjust][165]); it is a bound or dummy variable similar to the *x*
in the integral from 0 to 1 of *x* d*x*. By introducing this definition, we effectively extend the
syntax with new class expressions of the form (𝐴 ∪ 𝐵) that can be substituted for class variables.

The number 2 in [Figure 1][166] is another example of a defined class builder, in this case a
constant with no arguments. Our definition [df-2][167], 2 = (1 + 1), involves yet more defined class
constants (1 defined in [df-1][168] and + defined in [df-add][169]), so it is not obviously a class
builder at this point. But if we backtrack far enough through several definitions, we will
eventually end up with 2 = {𝑥 ∣...} where "..." is a wff in the extended language.

We still need to determine what the above examples correspond to in the basic language used by the
axioms. In essence, there is an algorithm to translate each wff containing class builders to a
unique equivalent wff in the basic language of predicate calculus used for set theory. Later in this
section, we will show an example using the algorithm to provide a feel for how to recover the basic
language from a wff in the extended language.

The class definitions The algorithm for eliminating class builders is accomplished using the
following three definitions, which in effect provide the recursive definition of wffs containing
class builders.


─────────────────────────────────────┬───────┬───────────────────────────
[Definition of class                 │df-clab│⊢ (𝑥 ∈ {𝑦 ∣ 𝜑} ↔ [𝑥 / 𝑦]𝜑) 
abstraction][170]                    │       │                           
─────────────────────────────────────┼───────┼───────────────────────────
[Definition of class equality][171]  │dfcleq │⊢ (𝐴 = 𝐵 ↔ ∀𝑥(𝑥 ∈ 𝐴 ↔ 𝑥 ∈  
                                     │       │𝐵))                        
─────────────────────────────────────┼───────┼───────────────────────────
[Definition of class membership][172]│df-clel│⊢ (𝐴 ∈ 𝐵 ↔ ∃𝑥(𝑥 = 𝐴 ∧ 𝑥 ∈  
                                     │       │𝐵))                        
─────────────────────────────────────┴───────┴───────────────────────────

Definition [df-clab][173] extends the ∈ connective, and thus the definition of a wff, to allow a
class builder on the right-hand side of the ∈ connective, instead of (previously) only setvar
variables on each side. Note that the 𝜑 in [df-clab][174] is a wff in the extended language and thus
may itself contain class builders. The proper substitution denoted by [𝑥 / 𝑦]𝜑 has been previously
defined by [df-sb][175].

Definition [df-cleq][176] extends the = connective to allow classes on both sides. The definition
itself has a hypothesis, and above we show the derived version [dfcleq][177] with the hypothesis
eliminated for practical use. Without its hypothesis, [df-cleq][178] would produce the axiom of
extensionality [ax-ext][179] as a special case and thus is not sound (is not conservative) in the
context of predicate calculus without set theory. If we assume our class theory definitionally
extends set theory rather than just predicate calculus, the hypothesis is unnecessary and merely
provides us with a somewhat nitpicking isolation from set theory. Most authors do not include this
hypothesis. An advantage of the hypothesis for us is that we must explicitly use [ax-ext][180] to
satisfy it as in Theorem [dfcleq][181], meaning we can more easily determine exactly where
[ax-ext][182] was used by a proof.

Definition [df-clel][183] similarly extends the ∈ connective to allow classes on both sides. We do
not need a hypothesis as we did for [df-cleq][184] because its instances are theorems of predicate
calculus (see Theorem [cleljust][185]).

As an important special case, a setvar variable 𝑥 can be considered a class because it equals the
class builder {𝑦 ∣ 𝑦 ∈ 𝑥} (Theorem [cvjust][186]). This is the justification for syntax builder
[cv][187], which for convenience allows us to substitute a setvar variable directly for a class
variable. (On the other hand, we may not substitute a class variable for a setvar variable
directly.)

An example To see an example of the algorithm, let us eliminate the class builder from the
(extended) wff {𝑥 ∣ 𝑥 = 𝑎} = 𝑏. For simplicity we will assume all setvar variable pairs are
[distinct][188]. Treating the setvar variable 𝑏 as a class and applying [dfcleq][189], we get ∀𝑦(𝑦 ∈
{𝑥 ∣ 𝑥 = 𝑎} ↔ 𝑦 ∈ 𝑏). Applying [df-clab][190], we get ∀𝑦([𝑦 / 𝑥]𝑥 = 𝑎 ↔ 𝑦 ∈ 𝑏). Applying
[df-sb][191], we get ∀𝑦((𝑥 = 𝑦 → 𝑥 = 𝑎) ∧ ∃𝑥(𝑥 = 𝑦 ∧ 𝑥 = 𝑎)) ↔ 𝑦 ∈ 𝑏), which is our final expression
in the basic language of predicate calculus (implicitly assuming we have expanded the defined
connectives ∃, ∧, ↔). This expression is unique provided we have some canonical convention for
traversing the extended wff syntax tree and for naming any new variables introduced at each step. By
the way, this can be simplified with additional applications of predicate calculus to become ∀𝑦(𝑦 =
𝑎 ↔ 𝑦 ∈ 𝑏).

Our example showed the elimination of class builders from a wff with no class variables. If the wff
also contains class variables, these can be converted to wff variables by substituting {𝑥 ∣ 𝜑} for
𝐴, {𝑦 ∣ 𝜓} for 𝐵, and so on, where 𝜑 and 𝜓 are new wff variables that don't occur in the original
extended wff. The variables 𝑥 and 𝑦 are arbitrary (and may be the same) as long as they don't
conflict with any distinct variable requirements imposed on 𝐴 and 𝐵. We then would follow the
procedure of the above example. The description of theorem [eqabb][192] goes into more detail and
gives some actual database examples where this translation takes place.

Justification The theory of classes can be shown to be an eliminable and conservative extension of
set theory.

The eliminability property shows that for every formula in the extended language we can build a
logically equivalent formula in the basic language; so that even if the extended language provides
more ease to convey and formulate mathematical ideas for set theory, its expressive power is not in
fact stronger than the basic language's expressive power. The conservation property shows that for
every proof of a formula of the basic language in the extended system we can build another proof of
the same formula in the basic system; so that, concerning theorems on sets only, the deductive
powers of the extended system and of the basic system are identical. Together, these properties mean
that the extended language can be treated as a definitional extension that is sound.

A rigorous justification, which we will not give here, can be found in [[Levy][193]] pp. 357-366,
supplementing his informal introduction to class theory on pp. 7-17. Two other good treatments of
class theory are provided by [[Quine][194]] pp. 15-21 and [[TakeutiZaring][195]] pp. 10-14. Quine's
exposition is nicely written and very readable. Our purple class variables are Greek letters in
[Quine], and he uses an archaic dot notation instead of parentheses, but this is a relatively minor
hurdle especially since you can compare the Metamath versions of many of the theorems.

History According to [[Levy][196]] (p. 11), the way we introduce classes was first explicitly stated
by [[Quine][197]] (1963), who built on ideas stemming from Paul Bernays, particularly in Bernays'
book *Axiomatic Set Theory* (1958). Quine uses the term "virtual classes" for the theory of classes
we use here. Also according to Levy (pp. 11, 6, and 87), the informal idea of using classes that
cannot belong to other classes occurred to Cantor in 1899 after he became aware of the [Burali-Forti
paradox][198] (1897) and the fact that Cantor's theorem [fails][199] for the universe V (1899).
([Russell's paradox][200] came later, in 1901.) There are other (different) axiomatic set theories,
including Russell's theory of types, New Foundations, Quine's previous "Mathematical Logic" system,
and von Neumann-Bernays-Gödel (NBG). A more detailed comparison of ZFC using Quine's virtual classes
with these other axiomatic set theories can be found in [[Quine][201]] (1963) pp. 15-21 and pp.
241-332.

Definitions or axioms? Levy presents our three definitions above as axioms rather than definitions.
There is nothing wrong with this in principle, but because they are conservative and eliminable,
they do not strengthen the underlying set theory. They just specify a mechanical transformation to
and from a different but equivalent notation.

On the other hand, they have a character that is somewhat different from "ordinary" definitions such
as [df-an][202] that simply introduce a new symbol (or unique symbol combination) to abbreviate a
longer string of symbols. All such ordinary definitions can be verified for soundness with a
relatively straightforward algorithm that is incorporated into the mmj2 program. But there is no
simple way to verify automatically the soundness of our three class definitions. Instead, we must
trust the relatively involved metamathematics of Levy's (or other authors') eliminability and
conservation proofs that exist only on paper, at least until a sufficiently sophisticated
definitional soundness checker becomes available. The requirement of such trust makes it tempting to
call them axioms, since that would relieve us of responsibility in the unlikely event that a flaw is
uncovered in one of the eliminability/conservation proofs.

Personally I feel that despite their formal complexity, the eliminability/conservation properties
are still in some sense "obvious," particularly after gaining some experience with examples such as
the above. Moreover, calling them axioms could be slightly misleading by suggesting that the axioms
of ZFC set theory are not sufficient. For these reasons I chose to call them definitions, at the
risk of slightly compromising the ideal of computer-verified perfect rigor. It is rather trivial in
the Metamath language to switch them to axioms if we want to achieve this rigor, since the
distinction between the two is merely a naming convention of "ax-" vs. "df-" prefixes; see the
discussion below on [definitions][203].

The specific reason that the current mmj2 definitional soundness checker will reject our three
definitions for classes is that they overload (and thus in a naive sense redefine) existing
connectives. There is one other definition in the set.mm database that mmj2 is unable to check:
[df-bi][204], which does not connect definiendum and definiens with ↔ or =. For these four cases, it
is assumed that we are satisfied with a soundness justification outside of Metamath or its tools,
and we specify them as exceptions in mmj2's RunParms.txt file:


        SetMMDefinitionsCheckWithExclusions,ax-*,df-bi,df-clab,df-cleq,df-clel

A skeptic is free to rename these four to be axioms (with prefix "ax-"), but currently we consider
the compromise acceptable. Moreover, the above mmj2 statement tells us exactly which definitions we
are trusting without computer verification, so we can easily determine exactly what is being
assumed.


A Theorem Sampler


*Set theory can be viewed as a form of exact theology.*
—Rudy Rucker

Listed here are some examples of starting points for your journey through the maze. You should study
some simple proofs from propositional calculus until you get the hang of it. Then try some predicate
calculus and finally set theory.

The [Theorem List][205] shows the complete set of theorems in the database. You may also find the
[Bibliographic Cross-Reference][206] useful. The [Metamath 100][207] list identifies the progress of
Metamath formalizations of the "top 100" theorems as tracked by Freek Wiedijk. Many people have
contributed; you can even see a [Youtube video showing a Gource visualization of Metamath set.mm
contributions over time][208] (and we hope that you'll become another contributor!).


──────────────────────────────────────────────┬─────────────────────────────────────────────────────
Propositional calculus                        │Set theory (cont.)                                   
                                              │                                                     
Law of identity [idALT][209]                  │The existence of omega (the gateway to "Cantor's     
                                              │paradise") [omex][234]                               
Peirce's axiom [peirce][210]                  │                                                     
                                              │Burali-Forti paradox [onprc][235]                    
Praeclarum theorema [anim12][211]             │                                                     
                                              │Transfinite induction [tfindes][236]                 
Law of excluded middle [exmid][212]           │                                                     
                                              │Transfinite recursion [tfr1][237] [tfr2][238]        
Proposition *5.18 from Principia Mathematica  │[tfr3][239]                                          
[pm5.18][213]                                 │                                                     
                                              │The amazing recursive definition generator           
The consensus theorem for logic circuits      │[df-rdg][240]                                        
[consensus][214]                              │                                                     
                                              │Schröder-Bernstein Theorem [sbth][241]               
Meredith's astonishing single axiom           │                                                     
[meredith][215]                               │Pigeonhole principle [php][242]                      
                                              │                                                     
Predicate calculus                            │Axiom of Infinity equivalent (neat!) [inf5][243]     
                                              │                                                     
Existential and universal quantifier swap     │Axiom of Choice equivalent (brainteaser!) [ac2][244] 
[19.12][216]                                  │                                                     
                                              │Zermelo's well-ordering theorem [weth][245]          
Existentially quantified implication          │                                                     
[19.35][217]                                  │Zorn's Lemma [zorn2][246]                            
                                              │                                                     
*x* = *x* [equid][218]                        │Generalized Continuum Hypothesis (GCH) equivalent    
                                              │[gch-kn][247]                                        
Definition of proper substitution [df-sb][219]│                                                     
                                              │Real and complex numbers ([27 postulates][248])      
Double existential uniqueness [2eu5][220]     │                                                     
                                              │Archimedean property of real numbers [arch][249]     
Set theory                                    │                                                     
                                              │Ordered pair theorem for non-negative integers       
Commutative law for union [uncom][221]        │[nn0opthi][250]                                      
                                              │                                                     
A basic relationship between class and wff    │The square root of 2 is irrational [sqrt2irr][251]   
variables [eqabb][222]                        │                                                     
                                              │The nesting of natural numbers, integers, rationals, 
Two ways of saying "is a set" [isset][223]    │reals, and complex numbers [nthruc][252]             
                                              │                                                     
Kuratowski's ordered pair definition and      │Complex number in terms of real and imaginary parts  
Theorem [df-op][224]                          │[replimi][253]                                       
                                              │                                                     
Russell's paradox [ru][225]                   │Triangle inequality for absolute value [abstrii][254]
                                              │                                                     
The value of a function [df-fv][226]          │Euler's identity e↑(i · π) = -1 [eulerid][255]       
                                              │                                                     
Cantor's theorem [canth][227]                 │There exist infinitely many primes [infpn][256]      
                                              │                                                     
Mathematical (finite) induction [findes][228] │The real numbers are uncountable [ruc][257]          
                                              │                                                     
Peano's postulates for arithmetic:            │                                                     
[peano1][229] [peano2][230] [peano3][231]     │                                                     
[peano4][232] [peano5][233]                   │                                                     
──────────────────────────────────────────────┴─────────────────────────────────────────────────────
2 + 2 = 4 Trivia


*We used to think that if we knew one, we knew two, because one and one are two. We are finding that
we must learn a great deal more about 'and.'*
—Sir Arthur Eddington


*First and above all he was a logician. At least thirty-five years of the half-century or so of his
existence had been devoted exclusively to proving that two and two always equal four, except in
unusual cases, where they equal three or five, as the case may be.*
—Jacques Futrelle, *Best "Thinking Machine" Detective Stories*


*This is a one line proof...if we start sufficiently far to the left.*
—Unknown

The complete proof of a theorem all the way back to axioms can be thought of as a tree of
subtheorems, with the steps in each proof branching back to earlier subtheorems, until axioms are
ultimately reached at the tips of the branches. An interesting exercise is, starting from a theorem,
to try to find the longest path back to an axiom. Trivia Question: What is the longest path for the
theorem 2 + 2 = 4?

[Orange julia butterfly with 2+2=4 dots. Credit: N. Megill 2004. Public domain.]

Trivia Answer: A longest path back to an axiom from 2 + 2 = 4 is 184 layers deep! By following it
you will encounter a broad range of interesting and important set theory results along the way. You
can follow them by drilling down this path. Or you can start at the bottom and work your way up,
watching mathematics unfold from its axioms. (Hint: hovering the mouse over the links below will
show their descriptions.)

[2p2e4][258] <- [2cn][259] <- [2re][260] <- [1re][261] <- [axrrecex][262] <- [recexsr][263] <-
[sqgt0sr][264] <- [pn0sr][265] <- [distrsr][266] <- [dmaddsr][267] <- [addclsr][268] <-
[addsrpr][269] <- [enrer][270] <- [addcanpr][271] <- [ltapr][272] <- [ltaprlem][273] <-
[ltexpri][274] <- [ltexprlem7][275] <- [ltaddpr][276] <- [addclpr][277] <- [addclprlem2][278] <-
[addclprlem1][279] <- [ltrnq][280] <- [dmrecnq][281] <- [recclnq][282] <- [recidnq][283] <-
[recmulnq][284] <- [mulassnq][285] <- [mulerpq][286] <- [nqereq][287] <- [nqercl][288] <-
[nqerf][289] <- [nqereu][290] <- [enqer][291] <- [mulcanpi][292] <- [nnmcan][293] <- [nnmword][294]
<- [nnmord][295] <- [nnmordi][296] <- [nnaword1][297] <- [nnaword][298] <- [nnaord][299] <-
[nnaordi][300] <- [nnasuc][301] <- [onasuc][302] <- [frsuc][303] <- [rdgsucg][304] <-
[rdgdmlim][305] <- [tfr1a][306] <- [tfrlem16][307] <- [tfrlem15][308] <- [tfrlem13][309] <-
[tfrlem12][310] <- [tfrlem11][311] <- [tfrlem10][312] <- [tfrlem7][313] <- [tfrlem5][314] <-
[tfrlem2][315] <- [tfrlem1][316] <- [fvreseq][317] <- [eqfnfv][318] <- [mpteqb][319] <-
[fvmpt2][320] <- [fvmpt2i][321] <- [fvmpti][322] <- [fvmptg][323] <- [fvopab3ig][324] <-
[funopfv][325] <- [funbrfv][326] <- [funeu][327] <- [funmo][328] <- [dffun6][329] <- [dffun6f][330]
<- [dffun3][331] <- [dffun2][332] <- [brcnv][333] <- [brcnvg][334] <- [opelcnvg][335] <-
[brabg][336] <- [brabga][337] <- [opelopabga][338] <- [copsex2g][339] <- [copsexg][340] <-
[eqvinop][341] <- [opth2][342] <- [opthg2][343] <- [opthg][344] <- [opth][345] <- [opeq1d][346] <-
[opeq1][347] <- [ifbieq1d][348] <- [ifeq1d][349] <- [ifeq1][350] <- [dfif6][351] <- [uneq12i][352]
<- [uneq12][353] <- [uneq2][354] <- [uncom][355] <- [uneqri][356] <- [elun][357] <- [elab2g][358] <-
[elabg][359] <- [elabgf][360] <- [vtoclgf][361] <- [issetf][362] <- [nfeq2][363] <- [nfeq][364] <-
[nfcri][365] <- [nfcrii][366] <- [hblem][367] <- [clelsb1][368] <- [sbco2][369] <- [sbied][370] <-
[sbie][371] <- [sbbi][372] <- [sban][373] <- [sbim][374] <- [sbi2][375] <- [sbn][376] <-
[dfsb3][377] <- [dfsb2][378] <- [sb4][379] <- [equs5][380] <- [axc15][381] <- [ax13a2][382] <-
[ax13v2][383] <- [dveeq2][384] <- [ax12lem3][385] <- [19.36v][386] <- [19.36][387] <- [19.9][388] <-
[19.9h][389] <- [19.9t][390] <- [19.9ht][391] <- [a10e][392] <- [axc7][393] <- [sp][394] <-
[19.8a][395] <- [equcoms][396] <- [equcomi][397] <- [equid][398] <- [eximii][399] <- [eximi][400] <-
[exim][401] <- [alnex][402] <- [con2bii][403] <- [con1bii][404] <- [xchbinx][405] <- [notbii][406]
<- [notbi][407] <- [notbid][408] <- [3bitr3g][409] <- [syl5bbr][410] <- [syl5bb][411] <-
[bitrd][412] <- [bitri][413] <- [sylibr][414] <- [biimpri][415] <- [bicomi][416] <- [bicom1][417] <-
[bi2][418] <- [dfbi1][419] <- [impbii][420] <- [bi3][421] <- [simprim][422] <- [impi][423] <-
[con1i][424] <- [nsyl2][425] <- [mt3d][426] <- [con1d][427] <- [notnot1][428] <- [con2i][429] <-
[nsyl3][430] <- [mt2d][431] <- [con2d][432] <- [notnot2][433] <- [pm2.18d][434] <- [pm2.18][435] <-
[pm2.21][436] <- [pm2.21d][437] <- [a1d][438] <- [syl][439] <- [mpd][440] <- [a2i][441] <-
[ax-2][442]

(The list above was produced by typing the commands "read set.mm" then "show trace_back 2p2e4
/essential /count_steps" in the [Metamath program][443], after replacing the references to the
[complex number axioms][444] to their corresponding theorems, such as "ax-rnegex" to "axrnegex". The
complex numbers are a special case where we restate as axioms the theorems that derive their
properties, in order to reduce clutter in the axiom and definition lists on the web pages.)

The complete proof of 2 + 2 = 4 involves 2,913 subtheorems including the 184 above. (The command
"show trace_back 2p2e4 /essential" will list them.) These have a total of 26,323 steps—this is how
many steps you would have to examine if you wanted to verify the proof by hand in complete detail
all the way back to the axioms of ZFC set theory. ([Later][445] we will compare Metamath's proof
length to that of formal proofs defined in logic textbooks.)

One of the reasons that the proof of 2 + 2 = 4 is so long is that 2 and 4 are complex numbers—i.e.
we are really proving (2+0i) + (2+0i) = (4+0i)—and these have a complicated construction (see the
[Axioms for Complex Numbers][446]) but provide the most flexibility for the arithmetic in our set.mm
database. In terms of textbook pages, the construction formalizes perhaps 70 pages from Takeuti and
Zaring's *Introduction to Axiomatic Set Theory* (and its predicate calculus prerequisite) to obtain
natural number (finite ordinal) arithmetic, followed by essentially all of Landau's 136-page
*Foundations of Analysis*.

We selected the theorem 2 + 2 = 4 for this example rather than 1 + 1 = 2 because the latter is
essentially the definition we chose for 2 and thus has a trivial proof.

In *Principia Mathematica*, 1 and 2 are *cardinal* numbers, so its proof of 1 + 1 = 2 is different:
see [dju1p1e2][447] for a translation into modern notation.


Appendix 1: A Note on the Axioms The Metamath axiom system defined in set.mm is founded on a
simplified formalization of predicate calculus with equality published by logician Alfred Tarski in
1965 (see [[Tarski][448]], system S2 mentioned in the last paragraph of p. 77; reproduced as system
S3 in Section 6 of [[Megill][449]]). Tarski's system is exactly equivalent to the [traditional][450]
textbook formalization, but (by clever use of equality axioms) it eliminates the latter's primitive
notions of "proper substitution" and "free variable," replacing them with direct substitution and
the notion of a variable not occurring in a formula (which we express with [distinct variable][451]
conditions).

In advocating his system, Tarski wrote, "The relatively complicated character of [free variables and
proper substitution] is a source of certain inconveniences of both practical and theoretical nature;
this is clearly experienced both in teaching an elementary course of mathematical logic and in
formalizing the syntax of predicate logic for some theoretical purposes" [[Tarski][452], p. 61].

Tarski's system was designed for proving specific theorems rather than more general theorem schemes
(which we will define below). However, theorem schemes are much more efficient than specific
theorems for building a body of mathematical knowledge, since they can be reused with different
instances as needed. While Tarski does derive some theorem schemes from his axioms, their proofs
require concepts that are "outside" of the system, such as induction on formula length. The
verification of such proofs is difficult to automate in a proof verifier. (Specifically, Tarski
treats the formulas of his system as set-theoretical objects. In order to verify the proofs of his
theorem schemes, a proof verifier would need a significant amount of set theory built into it.)

The Metamath axiom system (shown as [ax-1 through ax-13][453] above) extends Tarski's system to
eliminate this difficulty. The additional axiom schemes (as we will call them in this section; see
below) endow Tarski's system with a nice property we call scheme completeness (defined in Remark 9.6
of [[Megill][454]] and called "metalogical completeness" there). As a result, we can prove *any*
theorem scheme expressable in the "simple metalogic" of Tarski's system by using *only* Metamath's
direct substitution rule applied to the axiom system (and no other metalogical or set-theoretical
notions "outside" of the system). Simple metalogic consists of schemes containing wff metavariables
(with no arguments) and/or setvar (also called "individual") metavariables (i.e. in the form of the
[wff syntax][455] described above), accompanied by optional conditions, each stating that two
specified setvar metavariables must be distinct or that a specified setvar metavariable may not
occur in a specified wff metavariable. Metamath's logic and set theory axiom and rule schemes are
all examples of simple metalogic. The schemes of [traditional predicate calculus with equality][456]
are examples which are *not* simple metalogic, because they use wff metavariables with arguments and
have "free for" and "not free in" side conditions.

Axioms vs. axiom schemes An important thing to keep in mind is that Metamath's "axioms" and
"theorems" are not the actual axioms and theorems of first-order logic (i.e. the [traditional
predicate calculus][457] of textbooks) but instead should be interpreted as schemes, or recipes, for
generating those axioms and theorems. In particular, the (red) "setvar variables" in Metamath's
axioms are *not* the individual variables of the actual axioms; instead, they are metavariables
ranging over these individual variables (which in turn range over the individuals in the logic's
universe of discourse—in our case of set theory, the universe of discourse is the collection of all
sets). This can cause and has caused confusion, particularly because of the superficial resemblance
between the two. For clarity, we will refer to Metamath's axioms as axiom schemes in this section
and whenever we discuss Metamath in the context of first-order logic.

The *actual* language (also called the object language) of first-order logic consists of starting
(or primitive or atomic) wffs (well-formed formulas) constructed from actual individual variables
*v*1, *v*2, *v*3,... connected by = and ∈ (such as "*v*2 = *v*4" and "*v*3 ∈ *v*2"), which are then
used to build up larger wffs with the connectives →, ¬, and ∀ (such as "(¬ *v*2 = *v*4 → ∀ *v*6 *v*3
∈ *v*2)"). That is the complete language needed to express all of set theory and thus essentially
all of mathematics. That's it; there is nothing else! There are no other variable types in the
actual language; in particular *there are no variables representing wffs.* Each of our axiom schemes
corresponds to (generates) an infinite set of such actual formulas that match its pattern. For
example, "(∀ *v*1 ¬ *v*3 ∈ *v*2 → ¬ *v*3 ∈ *v*2)" is an actual axiom since it matches axiom scheme
[ax-c5][458], "(∀𝑥𝜑 → 𝜑)," when we substitute "*v*1" for "𝑥" and "¬ *v*3 ∈ *v*2" for "𝜑."

Even in propositional calculus, the "axioms" are really axiom schemes. Although logic textbooks may
have "propositional variables" that are manipulated directly and even treated as primitive for
convenience (or certain theoretical purposes), they are really wff metavariables when used as part
of the foundations for mathematics. An actual axiom of propositional calculus has no propositional
or wff variables but only the kinds of actual wffs we just described. Each axiom scheme is a
template corresponding to an infinite number of actual axioms. For example, neither the general form
of [ax-1][459] "(𝜑 → (𝜓 → 𝜑))" nor the more restrictive substitution instance "(𝑥 = 𝑦 → (𝑥 = 𝑦 → 𝑥 =
𝑦))" is an actual axiom—both are schemes ranging over an infinite number of actual axioms, with
fewer (in the proper subset sense) actual axioms in the second case, of course. An example of an
actual propositional calculus axiom is the instance "( *v*3 = *v*5 → (∀ *v*6 *v*1 ∈ *v*4 → *v*3 =
*v*5))."

Our axiom schemes vs. Tarski's original axiom schemes In the figure below we compare Tarski's
original axiom schemes for predicate calculus with equality (rightmost column) with the extensions
that provide our system with "[scheme completeness][460]."

─────────────┬───────────────────────────────────────┬───────────────────────────────────────
             │Metamath's axiom schemes               │Tarski's system S2                     
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-gen][461]│⊢ 𝜑 ⇒ ⊢ ∀𝑥𝜑                            │⊢ 𝜑 ⇒ ⊢ ∀𝑥𝜑                            
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-4][462]  │⊢ (∀𝑥(𝜑 → 𝜓) → (∀𝑥𝜑 → ∀𝑥𝜓))            │⊢ (∀𝑥(𝜑 → 𝜓) → (∀𝑥𝜑 → ∀𝑥𝜓))            
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-5][463]  │⊢ (𝜑 → ∀𝑥𝜑), where 𝑥 does not occur in │⊢ (𝜑 → ∀𝑥𝜑), where 𝑥 does not occur in 
             │𝜑                                      │𝜑                                      
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-6][464]  │⊢ ¬ ∀𝑥¬ 𝑥 = 𝑦                          │⊢ ¬ ∀𝑥¬ 𝑥 = 𝑦, where 𝑥 and 𝑦 are       
             │                                       │distinct                               
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-7][465]  │⊢ (𝑥 = 𝑦 → (𝑥 = 𝑧 → 𝑦 = 𝑧))            │⊢ (𝑥 = 𝑦 → (𝑥 = 𝑧 → 𝑦 = 𝑧))            
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-8][466]  │⊢ (𝑥 = 𝑦 → (𝑥 ∈ 𝑧 → 𝑦 ∈ 𝑧))            │⊢ (𝑥 = 𝑦 → (𝑥 ∈ 𝑧 → 𝑦 ∈ 𝑧))            
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-9][467]  │⊢ (𝑥 = 𝑦 → (𝑧 ∈ 𝑥 → 𝑧 ∈ 𝑦))            │⊢ (𝑥 = 𝑦 → (𝑧 ∈ 𝑥 → 𝑧 ∈ 𝑦))            
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-10][468] │⊢ (¬ ∀𝑥𝜑 → ∀𝑥¬ ∀𝑥𝜑)                    │                                       
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-11][469] │⊢ (∀𝑥∀𝑦𝜑 → ∀𝑦∀𝑥𝜑)                      │                                       
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-12][470] │⊢ (𝑥 = 𝑦 → (∀𝑦𝜑 → ∀𝑥(𝑥 = 𝑦 → 𝜑)))      │                                       
─────────────┼───────────────────────────────────────┼───────────────────────────────────────
[ax-13][471] │⊢ (¬ 𝑥 = 𝑦 → (𝑦 = 𝑧 → ∀𝑥𝑦 = 𝑧))        │                                       
─────────────┴───────────────────────────────────────┴───────────────────────────────────────

Our extensions [ax-10][472] through [ax-13][473] are provable (outside of Metamath) as metatheorems
of Tarski's original system, meaning that they are sound (and also logically redundant).
Specifically, all instances of our extensions not involving wff metavariables (and with all setvar
metavariables mutually distinct) can be proved inside of Metamath from Tarski's axiom schemes; see
theorem schemes [ax10w][474], [ax11w][475], [ax12w][476], and [ax13w][477]. These theorem schemes
have hypotheses that can be eliminated for such instances. Theorem [ax12wdemo][478] shows an example
of how we can eliminate the hypotheses of [ax12w][479]. Finally, we combine all such instances
(outside of Metamath) to result in the metatheorem represented by our extensions.

Except for [ax-6][480], our system includes all axiom schemes of Tarski's system. In the case of
[ax-6][481], Tarski's version is weaker because it includes a distinct variable condition. If we
wish, we can also weaken our version in this way and still have a system that is scheme-complete.
Theorem [ax6][482] shows this by deriving, in the presence of the other axioms, our [ax-6][483] from
Tarski's weaker version [ax6v][484]. However, we chose the stronger version for our system because
it is simpler to state and easier to use. (Technically, our [ax-7][485], [ax-8][486], and
[ax-9][487] are sub-schemes of a single higher-order scheme in Tarski's system, but these three
suffice to make our system complete. Breaking them out makes our metalogic simpler by avoiding the
need for a notion of substitution into an atomic wff.)

Substitution instances of schemes In Metamath, by default (i.e., when no distinct variable
conditions are present; see [below][488]) there are no restrictions on substitutions that may be
made into a scheme, provided we honor the metavariable types (wff variable or setvar variable). This
is called direct substitution, in contrast to a more complicated "proper substitution" used in
textbook predicate calculus. Consider, for example, axiom scheme [ax-6][489], which can be
abbreviated as "∃𝑥𝑥 = 𝑦" (theorem scheme [ax6e][490]), "there exists (∃) an 𝑥 such that 𝑥 equals 𝑦".
In traditional predicate calculus, the first argument (a variable) of the quantifier ∃ is considered
"bound" in the wff serving as its second argument (i.e., in the quantifier's "scope"), otherwise it
is "free". Substitutions must follow careful rules taking into account whether the variable is bound
or free at a given position. In the Metamath language, ∃ is merely a 2-place prefix connective or
"operation" that evaluates to a wff, taking a setvar variable as its first argument and a wff as its
second, with no built-in concept of bound or free. In [ax6e][491], we place no restriction on what
actual variables may be substituted for bound metavariable 𝑥 and free metavariable 𝑦. We can even
substitute them with the same variable, seemingly at odds with the traditional rule that free and
bound variables must not clash. (When we do need to prohibit certain substitutions, we use "distinct
variable" conditions, which we describe below, that apply to a theorem scheme as a whole without
consideration of whether a variable's occurrence is free or bound. This makes figuring out what
substitutions are allowed very simple.)

The expression "∃𝑥𝑥 = 𝑦" is just a recipe for generating an infinite number of actual axioms. In an
actual axiom such as "∃ *v*3 *v*3 = *v*2," symbols *v*2 and *v*3 always represent distinct variables
by definition, because they have different names. Another axiom that results from the recipe is "∃
*v*3 *v*3 = *v*3," which has a different meaning but is still perfectly sound. On the other hand,
when working with the *actual* axioms there is no rule that allows us to infer "∃ *v*3 *v*3 = *v*3"
from "∃ *v*3 *v*3 = *v*2": they are independent axioms generated by the scheme. In the actual logic,
the only rules of inference are the (infinite) specific instances of the [modus ponens][492] and
[generalization][493] schemes—for example, from "∃ *v*3 *v*3 = *v*2" we can infer "∀ *v*6∃ *v*3 *v*3
= *v*2" by generalization—and there is no substitution rule built in as a primitive notion.

The proper substitution rule of traditional predicate calculus is a by-product of the axiom schemes
that were chosen, and the rule is necessary to ensure that these schemes generate only correct
actual axioms. Metamath uses different axiom schemes that do not require proper substitution but
generate exactly the same actual axioms as traditional predicate calculus. (A price we pay with
Metamath is a larger number of axiom schemes.) In the actual axioms, there are no primitive concepts
of "free", "bound", "distinct", or even "substitution"—these are all metamathematical notions at the
scheme level.

See also the text in the section header for ["Logical redundancy of ax-10, ax-11, ax-12,
ax-13"][494] and technical notes 2 and 3 below.

Metamath is a metalanguage that describes first-order logic ...and is not itself the language of
first-order logic. But the "meta" aspect runs deeper than just the fact that its axioms represent
schemes.

Traditional presentations of first-order logic also use schemes to describe the axioms. However, a
substitution instance of one of their axiom schemes is intended to be one specific axiom of the
actual logic. Formal proofs are defined so that they involve only actual axioms, to result in actual
theorems. Often textbooks will derive theorem schemes to obtain more generally useful results, but
such derivations are understood to be outside of the logic itself and may use metalogical techniques
such as induction on formula length that are not part of the axiom system.

Metamath's key and very important difference is that an application of its substitution rule *always
creates a new scheme from an old one.* Metamath works *only* with schemes and *never* with actual
axioms or theorems. For example, the schemes "(𝑥 = 𝑦 → 𝑥 = 𝑦)" and "((𝜓 → 𝜑) → (𝜓 → 𝜑))" are two
substitution instances that we can infer from the scheme "(𝜑 → 𝜑)." These substitution instances are
new schemes in their own right, into which we can make further substitutions to specialize them
further if we wish.

The setvar and wff variables of the Metamath language are one level up, or "meta," from the point of
view of the logic that it describes. (Hence the name *Metamath*, meaning "metavariable math.") Each
"theorem" on our proof pages is a theorem scheme (metatheorem) that is a recipe for generating an
infinite number of actual theorems. In fact, the Metamath language cannot even express specific,
actual theorems such as "( *v*1 = *v*2 → *v*1 = *v*2)." We would have to do that outside of the
Metamath system. Ordinarily this is unnecessary; even logic textbooks work informally with theorem
schemes after they explain how the formal system is constructed. Metamath formalizes the process for
working directly with schemes and makes the necessary metalogic (its substitution rule) part of the
axiom system itself, so that no techniques outside of the axiom system are needed to derive its
theorem schemes.

[As an example of the savings we achieve by using only schemes, the [2p2e4][495] proof described in
[2 + 2 = 4 Trivia][496] above requires 26,323 proof steps to be proved from the axioms of logic and
set theory. If we were not allowed to use different instances of intermediate theorem schemes but
had to show a direct object-language proof, where each step is an actual axiom or a rule applied to
previous steps, the complete formal proof would have 73,983,127,856,077,147,925,897,127,298 (about
7.4 x 10^28 or 74 octillion) proof steps. See also the remarks for Theorem [1kp2ke3k][497]. Note:
this calculation ignores the expansion of statements that use class notation, which would require
extra steps.]

Distinct metavariables Sometimes we must place restrictions on the formulas generated by a scheme in
order to ensure their soundness, and we use [distinct variable][498] conditions for this purpose.
For example, the theorem scheme [dtru][499], ¬ ∀𝑥 𝑥 = 𝑦, has the restriction that metavariables 𝑥
and 𝑦 cannot be substituted with the same actual variable, otherwise we could end up with the
nonsense substitution instance "¬ ∀ *v*1 *v*1 = *v*1" which would mean "it is not true that every
object *v*1 equals itself." The substitution rule of Metamath ensures that distinct variable
conditions "propagate" from axiom schemes having such restrictions into theorem schemes using those
axiom schemes; in other words, distinct variable conditions in axiom schemes are inherited by
theorems whose proofs make use of those schemes.

Note that distinct variable conditions are *metalogical* conditions imposed on certain axiom and
theorem *schemes.* They have no role in the actual logic (object language), where two actual
variables with different names are distinct by definition—simply because they have different names,
which is what "distinct" means. Thus in an actual theorem generated by [dtru][500], such as "¬ ∀
*v*1 *v* 1 = *v*2," it would be redundant (and meaningless) to require that *v* 1 and *v*2 be
distinct. There is no danger of inferring from this the falsehood "¬ ∀ *v* 1 *v*1 = *v*1", because
there is no substitution rule in the actual language that lets us do so.

As we mentioned (three paragraphs earlier), the Metamath language itself cannot express actual
theorems such as "¬ ∀ *v*1 *v*1 = *v*2." Instead, the distinct variable condition on [dtru][501] is
enforced at the level of theorem schemes. In this example, we are not allowed to substitute the same
metavariable, say 𝑧, for both 𝑥 and 𝑦 whenever we reference [dtru][502] in a proof step.

Technical notes 1. For anyone interested, here are some technical notes on the [[Megill][503]]
paper. The claim in Remark 9.6, "C14' is omitted from S3' because it can be proved from the others
using only simple metalogic," is proved in Theorem [axc14][504]. Axiom C16' is also redundant—see
Theorem [axc16][505], whose proof was discovered in 2006 and not known at the time of the paper. The
somewhat strange-looking axiom C10' ([ax-c10][506]) was found to be equivalent to the simpler
[ax-6][507] after the paper was submitted; see Theorem [ax6fromc10][508]. In the proof of the scheme
(or "metalogical") completeness theorem, Theorem 9.7, some details that are stated without proof as
"provable in S3' with simple metalogic," but which are nonetheless are tricky, are proved by
theorems [sbequ][509], [sbcom2][510], and [sbid2v][511].

2. Some people find it counter-intuitive that [ax-6][512] combines two conceptually different axiom
schemes, one where 𝑥 and 𝑦 are always distinct variables (one bound, one free) and another which
really means "∃𝑥 𝑥 = 𝑥." Raph Levien calls this "bundling" (see ["Principal instances of
metatheorems"][513] [retrieved 14-Jul-2015]; related pages are [ Distinctors vs. binder][514]
[retrieved 18-Apr-2016] and [ Distinct variables][515] [retrieved 18-Apr-2016]). We chose an
axiomatization with maximally bundled axiom schemes, making them more general, fewer in number, and
easier to use. See also the text in the section header for [ "Logical redundancy of ax-10, ax-11,
ax-12, ax-13"][516].

3. Tarski used the bundled axiom scheme "∃𝑥 𝑥 = 𝑦" in one of his systems without requiring that 𝑥
and 𝑦 be distinct variables (see [[KalishMontague][517]], footnote on p. 81). In other words, he
bundled it like we do to make it more general and able to prove instances of "∃𝑥 𝑥 = 𝑥" without
requiring a separate axiom scheme. On the other hand, he required that 𝑥 and 𝑦 be distinct in this
scheme of his system S2 mentioned above, since instances of "∃𝑥 𝑥 = 𝑥" could be derived in another
way (see the proof of [equid1][518]). Tarski considered it better to include the distinct variable
condition since he wanted to show the weakest possible axiom scheme needed for completeness, even
though it made the statement of the scheme more complex.

4. An interesting feature of Metamath's axiom system is that it is *finitely axiomatized* in the
following strong sense: at any point in a proof, there are only finitely many axiom schemes in the
system from which the next step can be chosen, and whenever the choice is valid i.e. unifies, there
is a unique *most general theorem* that that results. This fact is exploited and illustrated in the
[Metamath Solitaire][519] applet: you are presented with exactly all possible allowed (unifiable)
choices, and when you choose one you are shown the most general theorem that results. This is in
sharp contrast to traditional predicate calculus (and even Tarski's system), where we must choose
from an infinite number of substitution possibilities in order to apply an axiom. In particular, ZFC
set theory also becomes finitely axiomatized in this strong sense, because the Replacement Scheme of
ZFC becomes a single "axiom" (or more precisely, a single scheme expressable in simple metalogic)
under Metamath's formalization. (Note that the terminology "finitely axiomatized" is also used in
the literature in a different way, to mean a theory with finitely many actual axioms on top of
predicate calculus, *without* counting the infinite number of axioms of predicate calculus itself.
For example, under that meaning, ZFC set theory is not finitely axiomatized because its Axiom of
Replacement is a scheme describing an infinite number of axioms.)

5. Interestingly, Raph Levien has shown (see Item 16 on the [Workshop Miscellany][520] page) that in
the actual logic, if we are given "∃ *v*3 *v*3 = *v* 3" as a hypothesis, it is impossible to infer
from it, in the absence of scheme [ax-6][521], even the obvious equivalent "∃ *v*4 *v*4 = *v*4."


Appendix 2: Traditional Textbook Axioms of Predicate Calculus with Equality

If you want to acquire a practical working ability in logic, it is a good idea to become familiar
with the traditional textbook axioms. Their built-in concepts of free and bound variables and proper
substitution are a little more complex than Metamath's simple substitution rule and concept of two
variables being [distinct][522], but they allow you to work at a somewhat higher level and are
better suited than Metamath's for humans to understand intuitively. In fact, many of the proofs here
were sketched out informally using traditional methods before being formalized with Metamath's
axioms.

For classical propositional calculus, the traditional axiom and rule schemes are exactly the same as
Metamath's axioms and rule (interpreted as [schemes][523]), namely [ax-1][524], [ax-2][525],
[ax-3][526], and rule [ax-mp][527]. The additional axiom and rule schemes for traditional predicate
calculus with equality are the following. The first three are the axiom and rule schemes for
traditional predicate calculus, and the last two are the axiom schemes for the traditional theory of
equality. We link to the approximately equivalent theorem schemes in the Metamath formalization.


─────────────┬────────────────────────────────────────────────────────────────
[stdpc4][528]│⊢ ∀𝑥𝜑(𝑥) → 𝜑(𝑦), provided that 𝑦 is free for 𝑥 in 𝜑(𝑥)          
─────────────┼────────────────────────────────────────────────────────────────
[stdpc5][529]│⊢ ∀𝑥(𝜑 → 𝜓) → (𝜑 → ∀𝑥𝜓), provided that 𝑥 is not free in 𝜑       
─────────────┼────────────────────────────────────────────────────────────────
[ax-gen][530]│⊢ 𝜑 ⇒ ⊢ ∀𝑥𝜑                                                     
─────────────┼────────────────────────────────────────────────────────────────
[stdpc6][531]│⊢ ∀𝑥 𝑥 = 𝑥                                                      
─────────────┼────────────────────────────────────────────────────────────────
[stdpc7][532]│⊢ 𝑥 = 𝑦 → (𝜑(𝑥, 𝑥) → 𝜑(𝑥, 𝑦)), provided that 𝑦 is free for 𝑥 in 
             │𝜑(𝑥, 𝑥)                                                         
─────────────┴────────────────────────────────────────────────────────────────

Three of these axiom schemes have informal English-language conditions on them. These conditions are
somewhat awkward to formalize (see for example [[Tarski][533]] pp. 63-67), but they are not hard to
grasp intuitively. You can look them up in any elementary logic book. For example, they are
explained in Hirst and Hirst's [*A Primer for Logic and Proof*][534] [retrieved 17-Sep-2017] (PDF,
0.5MB); Section 2.4 (pp. 36-37; add 6 for the PDF page number) defines "free variable," and Section
2.11 (pp. 48-51) defines "free for." See pp. 15, 51, & 64 for the propositional calculus, predicate
calculus, and equality axioms respectively.

Stefan Bilaniuk's [*A Problem Course in Mathematical Logic*][535] [retrieved 21-Dec-2016], another
free on-line book, provides a more extensive study of logic.

Even though the traditional axiom system looks different from Metamath's, the two axiom systems
generate exactly the same set of *actual* theorems. (Read about [schemes][536] in the previous
section to understand what "actual" means.) Specifically, all of Metamath's axiom schemes [ax-4
through ax-13][537] can be proved as theorem schemes of the traditional system. Conversely, every
instance of the traditional axiom schemes is an instance of a theorem scheme provable from
Metamath's axiom schemes. Because people familiar with the traditional system have sometimes felt
there is something wrong with ours, particularly since it dispenses with the alpha conversion
required by proper substitution in the traditional axiom system, this bears repeating: Our axiom
schemes generate precisely the same object-language axioms and rules, no more and no fewer, as the
traditional schemes (Theorem 5 of [[Tarski][538]], p. 78).

Although in some sense the traditional axiom schemes are more compact than Metamath's [ax-4 through
ax-13][539] (4 schemes instead of 10), their purpose is simply to provide recipes for generating
actual axioms, from which we then prove actual theorems. Theorem schemes can also be proved from the
traditional axiom schemes, but their proofs use informal metalogical techniques that are not part of
the axiom system. In Metamath, on the other hand, we deal *only* with schemes and never with actual
axioms, and its formalization is designed to let us prove directly all possible theorem schemes of a
certain kind (specifically, all possible schemes expressible in [simple metalogic][540]).

Metamath's system does not have the traditional system's (metalogical) notions of "not free in" and
"free for" built in, so the traditional system's schemes cannot be translated exactly to schemes in
the Metamath language. However, we can emulate these notions (in either system, actually, since
every scheme in Metamath's system *is* a scheme of the traditional system) with conventions that are
based on a formula's *logical* properties (rather than its *structure,* as is the case with the
traditional axioms).

To say that "𝑥 is effectively not free in 𝜑," we can use the hypothesis 𝜑 → ∀𝑥𝜑. This hypothesis
holds in all cases where 𝑥 is not free in 𝜑 in the traditional system (and in a few other cases as
well, which is why we say "effectively"—for example, the wff 𝑥 = 𝑥 will satisfy the hypothesis, as
shown by Theorem [hbequid][541], even though 𝑥 is technically free in it). Because the idiom for
"effectively not free in" is so frequently used, we define a special logical connective, Ⅎ𝑥𝜑, that
stands for ∀𝑥(𝜑 → ∀𝑥𝜑). See Definition [df-nf][542].

We can emulate "free for" through the use of our definition of proper substitution [df-sb][543], as
shown for example in the approximate equivalent to [stdpc4][544] in the table above.

Both our system and the traditional system are called Hilbert-style systems. Two other approaches,
called natural deduction and Gentzen-style systems, are closely related to each other and embed the
[ deduction (meta)theorem][545] into its axiom system. Natural deduction is straightforward to
emulate in our Hilbert-style system by exploiting the properties of wff metavariables, using the
rules described on the [deduction form and natural deduction][546] page.


Appendix 3: Distinct Variables In logic and set theory literature, theorems (or more precisely,
theorem schemes) are often accompanied by side conditions, or provisos, written in informal English.
Typically, these are written after the statement of the theorem and expressed in a form such as
"where 𝑥 is a variable that [satisfies some constraint]". In order to satisfy the metalogical
requirements of [Tarski's axiom system][547] on which it is based, Metamath implements three kinds
of provisos through the use of its "distinct variable" conditions. These restrict what substitutions
are allowed, and often you will see a condition such as the following accompanying an axiom or
theorem (for example, [ax-5][548] and [dtru][549]):


Distinct variable groups: 𝑥, 𝑦 𝑥, 𝜑 𝑥, 𝐴

These three groups (in this example, three pairs) have the following meanings, respectively, as side
conditions of the theorem scheme or axiom scheme shown above them:

1. where 𝑥 and 𝑦 may not be substituted with the same setvar variable,
2. where whatever setvar variable is substituted for 𝑥 may not appear in the wff expression
   substituted for 𝜑, and
3. where whatever setvar variable is substituted for 𝑥 may not appear in the class expression
   substituted for 𝐴.
For brevity we may say more informally,

1. where 𝑥 and 𝑦 are distinct,
2. where 𝑥 does not occur in 𝜑, and
3. where 𝑥 does not occur in 𝐴.

[Actually, there is only *one* rule in the Metamath language itself: the two expressions that are
substituted for the two variables in a distinct variable pair must not have any variables in common.
This is why a distinct variable pair is officially called a "disjoint variable pair" in the Metamath
specification. We present the rule as three separate cases above for clarity. In the set theory
database file, set.mm, we adopt the convention that at least one setvar variable should always
appear in a distinct variable pair, so these are the only cases you will see. Under this convention,
a distinct variable pair such as "𝐴, 𝜑" will never occur, even though technically it is not
illegal.]

Note that


Distinct variable group: 𝜑, 𝑥 means the same thing as


Distinct variable group: 𝑥, 𝜑

Finally, if more than two variables appear in a group, this is an abbreviated way of saying that the
conditions apply to all possible pairs in the group. So,


Distinct variable group: 𝑥, 𝑦, 𝜑 means the same thing as


Distinct variable groups: 𝑥, 𝑦 𝑥, 𝜑 𝑦, 𝜑

The basic rule to remember is that distinct variable conditions are inherited by substitutions (that
take place in a proof step). For example, look at Step 1 of the proof of [cleljustALT][550], which
has a substitution instance of [ax-5][551]. As you can see, [ax-5][552] requires the distinct
variable pair 𝑥, 𝜑. Step 1 substitutes 𝑧 for 𝑥 and 𝑥 ∈ 𝑦 for 𝜑. This substitution transforms the
original distinct variable requirement into the two distinct variable pairs 𝑧, 𝑥 and 𝑧, 𝑦, which
will implicitly accompany Step 1 (although not shown explicitly in step 1 of the proof listing).
Thus, Step 1 can be read in full, "⊢ (𝑥 ∈ 𝑦 → ∀𝑧𝑥 ∈ 𝑦) where 𝑧, 𝑥 are distinct and 𝑧, 𝑦 are
distinct". The proof verifier will check that this requirement accompanies the final theorem,
otherwise it will flag the proof step as invalid. You can see this requirement in the "Distinct
variable groups" list on the [cleljustALT][553] page.

This can be confusing if you do not understand how distinct variables conditions are inherited from
referenced axioms or theorems in order to satisfy their distinct variable conditions. Let us
consider an example (courtesy of Gérard Lang). Naively, one might think that [ax-13][554] (which has
no distinct variable conditions) is derivable from [ax-5][555], arguing as follows. Since the letter
𝑥 has no occurrence in the wff 𝑦 = 𝑧, one might think that a direct application of [ax-5][556] would
give (𝑦 = 𝑧 → ∀𝑥 𝑦 = 𝑧) as a theorem with no distinct variable conditions, so that [ax-13][557]
easily follows by adding an antecedent with [a1i][558]. However, "distinct" does not *merely* mean
that two setvar variables have different names; it means that we also must prevent them from being
substituted with the same setvar variable in the future. (To be precise, they must represent object
language variables with different names; see [Axioms vs. axiom schemes][559] and [Distinct
metavariables][560] above.) To apply [ax-5][561], we must explicitly ensure that 𝑥 is distinct from
both 𝑦 and 𝑧 in 𝑦 = 𝑧 by adding two distinct variable conditions. Otherwise, we could further
substitute 𝑥 for 𝑦 to arrive at (𝑥 = 𝑧 → ∀𝑥𝑥 = 𝑧); this violates the distinct variable condition of
[ax-5][562] since 𝑥 *does* appear in 𝑥 = 𝑧. In fact, from [ax-5][563] we can conclude only the very
restricted Theorem [ax13w][564], which because of its distinct variable conditions is much weaker
than Axiom [ax-13][565]. (If we omit them, the Metamath proof verifier will give an error message.)
We say that the distinct variable conditions in [ax13w][566] are inherited from [ax-5][567].

In the Metamath language, distinct variable conditions are specified with $d statements, placed
before an assertion ($a or $p) and in the same scope. Each theorem must be accompanied by those $d
statements needed to satisfy all distinct variable conditions implicitly attached the proof steps.

To get a concrete feel for distinct variable conditions, you can temporarily comment out the "$d x z
$." condition for Theorem [cleljustALT][568] in the database file set.mm. When you try to verify the
proof with the [Metamath program][569], the resulting error message will read as follows. (Note that
the "normal" proof format is used, which you can convert to with "save proof cleljustALT /normal",
and step 25 here corresponds to step 1 on the web page; steps 1–24 are syntax building steps that
can be seen with "show proof cleljustALT /all" while step 25 is the first essentiel step.)

MM> verify proof cleljustALT
cleljustALT
?Error at statement 5819, label "cleljustALT", type "$p":
      wel vz ax-5 vz vx vy elequ1 equsexhv bicomi $.
             ^^^^
There is a disjoint variable ($d) violation at proof step 25.  Assertion "ax-5"
requires that variables "ph" and "x" be disjoint.  But "ph" was substituted
with "x e. y" and "x" was substituted with "z".
Variables "x" and "z" do not have a disjoint variable condition in the
assertion being proved, "cleljustALT".
Such error messages can also be used to determine any missing $d statements during proof
development. Alternately, the [mmj2][570] program will compute the necessary $d's automatically.

For a dynamic example of distinct variable inheritance in action, enter the [first proof][571] of 𝑥
= 𝑥 into the Metamath Solitaire applet, which automatically computes the distinct variable
conditions needed for a theorem being proved. Watch the transformation of the distinct variable
condition that is inherited at the tenth step (ax-16).

Constants are considered distinct from all variables and never appear (nor are allowed to appear) in
a distinct variable group. See the comment for [isset][572].


History of distinct variables

The idea of using distinct variable conditions in place of the more complicated free-variable
concept of [traditional predicate calculus][573] was first stated by Tarski in 1951, with proofs
published in 1965 [[Tarski][574], p. 61, footnote]. It also allows us to dispense with proper
substitution as a primitive notion in the axiom system: "Instead of the general notion of proper
substitution we use...a much more elementary and special notion: that of replacement, of one
variable by another, in an atomic formula." [[Tarski][575], p. 62].

Below we show the two axiom schemes of Tarski's system that involve distinct variable conditions, in
their original form:

──────────────────────────────────────────────────────────────────────────────────────
[Tarski's axiom schemes with distinct variable conditions]                            
──────────────────────────────────────────────────────────────────────────────────────
Tarski's original axiom schemes with distinct variable conditions [[Tarski][576], p.  
75]                                                                                   
──────────────────────────────────────────────────────────────────────────────────────

Tarski uses "∧" and "≡" in place of our "∀" and "=" respectively (which allows him to use "=" for
the side condition of (B7)), and the notation "OC(*φ*)" means "the set of all variables occurring in
*φ*". These two schemes are identical to our [ax-5][577] and [ax6v][578], which are accompanied by
distinct variable conditions that can be read "where 𝑥 doesn't occur in 𝜑" and "where 𝑥 and 𝑦 are
distinct" respectively. (Our official [ax-6][579] does not have a distinct variable condition in
order to make it simpler to state and use.)


Notes on distinct variables

Note 1 Sometimes new or "dummy" variables are used inside of a proof that do not appear in the
theorem being proved. On the web pages we omit them from the "Distinct variable group(s)" list,
which is intended to show only those distinct variable conditions that need to be satisfied by any
proof that *references* the theorem.

If a proof uses dummy variables, we list them on a separate line beginning "Dummy variables...". For
simplicity, we assume that dummy variables are mutually distinct and distinct from all other
variables, and this assumption is indicated after the list.

Sometimes it is not strictly necessary for a dummy variable to be distinct from certain other
variables, for example when those variables do not participate in the part of the proof using the
dummy variable. In that case, it is *optional* whether or not we include those variables paired with
the dummy variable in $d statements in the set.mm file. Sometimes you will see such optional $d
statements omitted, depending on the preferences of the person who created the proof.

In principle, we could list on the web page only those pairs with dummy variables that are actually
specified to be distinct in the set.mm file. However, that would make the "Dummy variable" list very
long for some theorems with many variables, full of gaps and hard to read. (We used to do something
like that, and people didn't like it.) So for simplicity, we just assume that all possible variable
pairs with dummy variables are distinct, which is a conservative assumption that will always work.
This assumption makes it easiest to follow the proof since you don't have to keep checking a
distinct variable list. If you want to see which ones were actually specified for a particular
proof, you can consult the set.mm source file or use the metamath program command 'show statement
... /full'.

Note 2 An issue that has been debated is whether the Metamath language specification should be
changed so that $d statements associated with dummy variables are assumed implicitly. This is partly
a matter of taste, but so far I have felt it better to require explicit $d statements for them.
There are good arguments both ways, but to me, philosophically it makes the language more
transparent, if more tedious. Beginners may find an explicit list helpful for understanding proofs,
since nothing is hidden. Making it optional would complicate the Metamath spec slightly, since it
would require an exception added to $d verification. If we want to use a proof step as a theorem
(such as when breaking a proof into lemmas), its subproof may fail if the step has previously dummy
variables whose $d statements are hidden.

Some people who dislike this requirement have made $d statements for dummy variables optional for
their Metamath language verifiers (although strictly speaking this violates the current Metamath
spec); an example is [Hmm][580]. There is nothing wrong with this in principle, and we could even
make *all* $d statements for theorems optional—see [Note 5][581] below.

Note 3 Textbooks often use the notation 𝜑(𝑥) to denote that variable 𝑥 may appear in a wff
substituted for 𝜑. The Metamath language doesn't have a notation like this, but we can achieve the
logical equivalent simply by *omitting* the distinct variable group 𝑥,𝜑, as the example [axsep][582]
shows.

We can reconstruct the 𝜑(𝑥) notation from the distinct variable conditions that are omitted. On the
individual web pages, this reconstruction for wff and class variables is shown under the "Distinct
variable groups" list and called "Allowed substitution hints". The notation 𝜑(𝑥) there means that 𝑥
may appear (free, bound, or both) in any expression substituted for wff variable 𝜑. Note that this
is an informal, unofficial notation, not part of the Metamath language.

Keep in mind that the "Allowed substitution hints" are *not* necessarily a complete list of all
possible substitutions but are provided as a convenience to help you more easily determine what
substitutions are allowed, in contrast to the "Distinct variable groups" which tell you which ones
are forbidden. There are six things to be aware of.

1. If the theorem has no "Distinct variable groups", such as [mpteq2ia][583], any substitution at
all (honoring the variable types) is permissable, so an "Allowed substitution hints" list would be
pointless and is not shown to reduce possibly-confusing clutter.

2. If the theorem has "Distinct variable groups" and a wff or class variable is missing from the
"Allowed substitution hints", such as the 𝐴 in [copsexg][584], it means (in this case) that neither
of the setvar variables 𝑥 or 𝑦 may appear in a class expression substituted for 𝐴. It is acceptable
to substitute any class expression for 𝐴 that *doesn't* contain these two variables but possibly
contains others such as 𝑧.

3. If the theorem has "Distinct variable groups" but does not have an "Allowed substitution hints"
list, such as [dftr5][585], it means that none of the setvar variables in the theorem or its
hypotheses may appear in expressions substituted for its wff or class variables; in this case,
neither 𝑥 nor 𝑦 may appear in an expression substituted for 𝐴. Other setvar variables such as 𝑧 may
appear, though. This includes any dummy variables that may be used by the proof, since they do not
appear in the theorem or its hypotheses.

4. The list of variables in parentheses after a wff or class variable shows the permissable setvar
variables that may appear in a substitution *among those in the theorem* (or its hypotheses). There
is nothing preventing the use of setvar variables that do not appear in the theorem. For example, in
[axsep][586], 𝑥 and 𝑤 may appear in an expression substituted for 𝜑, but not 𝑦 or 𝑧.

5. The setvar variable list in parentheses says nothing about whether those setvar variables need to
be mutually distinct. Only the "Distinct variable groups" list provides this information. For
example, 𝜑(𝑥,𝑦) appears in the "Allowed substitution hints" for both [ralcom][587] and
[ralcom2][588]; 𝑥 and 𝑦 must be distinct in the former but not the latter.

6. The hypotheses of a theorem may impose additional conditions on how the variables in parentheses
may appear in the wff or class variable. For example, in [vtoclef][589], although 𝑥 may appear in
the expression substituted for 𝜑, it must appear in a way that allows the first hypothesis Ⅎ𝑥𝜑 to be
satisfied. Thus 𝑥 usually can't occur as a free variable in 𝜑 because the first hypothesis won't be
satisfied. This is in contrast to informal textbook usage where 𝜑(𝑥) typically means 𝑥 occurs as a
free variable in 𝜑. The way to tell whether the meaning is "may occur free or bound" or "may occur
if bound by a quantifier" is to look for a hypothesis of this form.

Note 4 Distinct variable conditions are sometimes confusing to Metamath newcomers. Unlike
traditional predicate calculus, Metamath does not confer a special status to a bound variable (e.g.
the quantifier variable 𝑥 in ∀𝑥𝜑); there is no built-in concept of its "scope." All variables,
whether free or bound, are subject to the same direct substitution rule. Metamath's substitution
rule does not treat a variable after the quantifier symbol "∀" any differently from a variable after
any other constant connective such as "¬". The only thing that matters is that the syntax is legal
for the variable type (wff, setvar, or class), and that any distinct variable conditions are
satisfied. This different paradigm may take some getting used to by someone used to traditional
"proper substitution" (which involves analyzing the scopes of bound variables and renaming them if
necessary), even though it is significantly *simpler* than the traditional approach. (Indeed, as
described above, this simplicity was a primary motivation for Tarski's predicate calculus that
Metamath's set.mm axioms are based on.) Unless otherwise restricted by a distinct variable
condition, a quantified (or any other) variable is by default *not necessarily distinct* from other
variables in the same expression, whether bound or not. This is opposite the usual implicit
assumption in traditional mathematics. For example, in many textbooks, it would be implicit that the
two variables in theorem scheme [dtru][590] must be distinct (particularly since one is bound and
the other is free), but in Metamath this requirement must be made explicit. (On the other hand, in
an *instance* of dtru in the *actual* logic, which Metamath cannot express directly, the two
variables are implicitly distinct by definition, as explained the ["Distinct variables"][591]
subsection of Appendix 1 above.)

Note 5 Interestingly, the distinct variable conditions ($d statements) accompanying theorems are
theoretically redundant, because the proof verifier could in principle infer and propagate them
forward from the distinct variable conditions of the axioms. The [Metamath Solitaire][592] applet
does this, inferring both the resulting proof steps and their distinct variable conditions as you
feed it axioms to build a proof. (The [mmj2][593] program will also infer the necessary $d
statements for a proof under development.) The Metamath language spec, on the other hand, requires
them to be explicit partly to speed up the verifier (for example, otherwise we couldn't verify a
proof in isolation but would have to analyze every proof it depends on), but making them explicit
also allows someone writing a proof to easily determine by inspection the distinct variable
conditions of a theorem he or she wishes to reference.

Note 6 Introducing unnecessary distinct variable groups in a theorem is always allowed and doesn't
affect its proof in any way; all it does is weaken the theorem so that later theorems may also have
to be weaker in order to make use of it. We rarely do this, but when we do (for the purpose of
creating a weaker starting axiom), it has confused some people. I put a comment explaining this in
[ax6v][594], which is an example of such a weakening.

Note 7 Is it possible to do mathematics without using distinct variables (in any form, including
their implicit use in the traditional "not free in" and "free for" concepts)? The answer is a
qualified yes, but with some complications; see the discussion about [distinctors][595].

A curiosity Two otherwise identical theorems with different distinct variable conditions can express
different mathematical facts. Compare, for example, [dvdemo1][596] and [dvdemo2][597].

Appendix 4: A Note on Definitions

The bottom line The Metamath language itself doesn't have a separate mechanism for introducing
definitions and in particular ensuring their soundness. The only way to add a definition to a
database is to introduce it as a new axiom. In the same way that an axiom system can be
inconsistent, an unsound definition may lead to an inconsistency.

If you don't know what we mean by an unsound definition, here is a simple example. Suppose we modify
[df-2][598] to become the self-referential expression "2 = ( 1 + 2 )" instead of its present "2 = (
1 + 1 )". From this, we can easily prove the contradiction 0 = 1. Therefore, this "definition" leads
to inconsistency and is unsound. But since it is an axiomatic ($a) assertion in the database, the
Metamath proof verifier is perfectly content to use it as a new fact and does not issue an error
message.

Thus, the soundness of definitions must be verified independently of the Metamath proof verifier,
either by hand or through the use of a separate automated tool customized for the logic used by the
database.

For the specific case of the set theory database set.mm, Mario Carneiro added an enhancement to the
[mmj2][599] program that will verify the soundness of all but four definitions. There is a
specification for this definition checker in the "additional rules for definitions" section of the
[conventions][600] page. The mmj2 definition checker can be turned on by adding

> SetMMDefinitionsCheckWithExclusions,ax-*,df-bi,df-clab,df-cleq,df-clel

to the RunParms.txt file. (The 4 excluded definitions, [df-bi][601], [df-clab][602], [df-cleq][603],
and [df-clel][604], need to be justified with metalogic outside of mmj2's capabilities. You can
consider them additional axioms if it bothers you. See also the discussion "Definitions or Axioms?"
in the [Theory of Classes][605] section above.)

Discussion The Metamath language provides a very simple and general framework for describing
arbitrary formal systems. Intrinsically it "knows" nothing about logic or math; all it does is
blindly assure us that we cannot prove anything not permitted by whatever formal system we have
described to it. What we call an "axiom" is to Metamath merely a new pattern for combining symbols
that we have told it is permissable. The same holds for what we call "definitions"—to Metamath,
definitions merely extend the formal system with new patterns and are indistinguishable from axioms
by the proof verifier.

For convenience, we usually make an artificial distinction by prefixing definition names with "df-".
But every definition is assumed to have been metalogically justified *outside* of the Metamath
formalism as being sound. A definition is sound provided (1) it is eliminable (the wff for any
theorem containing a defined symbol can be converted to an equivalent wff without it) and (2) it is
conservative (a theorem should be provable from the original axioms after the definition is
eliminated).

This method of introducing definitions as new axioms keeps the Metamath language simple and not tied
to any specific formal system. A definition that is sound in one formal system may be unsound in
another. For example, a recursive definition may not be eliminable in a system too weak to prove the
necessary recursion theorem (and even when it is, its elimination can be quite complex; see e.g. the
discussion of [df-rdg][606]).

One extreme viewpoint is to consider all definitions to *be* additional axioms, following logicians
such as Leśniewski [C. Lejewski, "On implicational definitions," *Studia Logica* 8:189-208 (1958)].

> Leśniewski regards definitions as theses of the system. In this respect they do not differ either
> from the axioms or from theorems, i.e. from the theses added to the system on the basis of the
> rule of substitution or the rule of detachment. Once definitions have been accepted as theses of
> the system, it becomes necessary to consider them as true propositions in the same sense in which
> axioms are true. *[p. 190]*

This was roughly the original idea when Metamath was first designed. The concept of definitional
soundness was considered outside of the scope of the proof verification engine, because it is
intrinsically dependent on the strength of the axiom system. Instead, definitional soundness was
expected to be verified independently, either by hand or by an automated tool customized for the
particular logic system used by the database. (Such a tool might recognize the "df-" prefix as a
keyword meaning "definition" and would impose constraints to guarantee that the definition is sound
under the axiom system of that database.)

A non-mathematical (human) issue with a definition is whether it matches the generally understood
meaning of a concept. For example, if we swapped the symbols 4 ([df-4][607]) and 5 ([df-5][608])
everywhere, all definitions would remain sound, but we could "prove" that ( 2 + 2 ) = 5. For this
reason, all definitions still must be carefully scrutinized by a competent mathematician, and
obviously there is no way to automate this inspection.

The [definition list (3MB)][609] shows all of the definitions in the database. The web page for each
proof lists all definitions that were used somewhere along its path back to the axioms, so that
nothing is hidden from the user in this sense.

* (Thanks to Raph Levien, Freek Wiedijk, Bob Solovay, and Mario Carneiro for discussions on this
topic.) *


Appendix 5: How to Find Out What Axioms a Proof Depends On At the end of each proof there is a list
called "This theorem was proved from these axioms." These are the axioms needed to prove the theorem
from scratch, if all definitions were eliminated. (You can also do this from within the [Metamath
program][610]: after "read set.mm", type "show trace_back tfr2 /essential /axioms" to get the list
for Transfinite Recursion Theorem [tfr2][611].)

You shouldn't expect to see immediately the relationship between the axioms and the proof you are
looking at—it is typically not obvious at all. This list of axioms was determined by scanning back
through the proof tree until axioms were reached, and in fact required a considerable amount of CPU
power. Mathematicians generally like to prove theorems using as few axioms as possible, which we
also usually try to do, so this list is often the minimum axiom set needed for the proof you are
seeing.

For example, we see that the Axiom of Choice [ax-ac][612] was not needed to prove [tfr2][613], but
we did need the Axiom of Replacement [ax-rep][614].

In order to find out all theorems in the path(s) from [tfr2][615] back to the Axiom of Replacement,
type "show trace_back tfr2 /to ax-rep". The following list results, with the theorems in the order
that they occur in set.mm:

> [ax-rep][616] [axrep1][617] [axrep2][618] [axrep3][619] [axrep4][620] [funimaexg][621]
> [resfunexg][622] [fnex][623] [funex][624] [tfrlem14][625] [tfr1][626]

This is particularly useful when "debugging" a proof that we think shouldn't require a particular
axiom. It will let us identify where exactly the undesired axiom snuck in, so that the proof can be
modified to avoid it if possible.

Sometimes a proof becomes much longer if we want to avoid a certain axiom. For example, Theorems
[ondomon][627] and [hartogs][628] are the same except for their proofs. Compare the relatively short
proof of [ondomon][629], which uses [ax-ac2][630], with the much longer proof of [hartogs][631] and
its lemmas needed to avoid [ax-ac2][632].


Appendix 6: Notation for Function and Operation Values In general I tried to choose notation that is
conventional or familiar, but I had to weigh it against introducing ambiguity or making the
collection of notations too bloated. Traditional notation includes many prefix and infix operations;
parsing this in an unambiguous way can be complicated. Sometimes I favored simplicity and
consistency over convention. An important example is the notation for a function value [cfv][633],
expressed by "(𝐹‘𝐴)". Here "𝐹" is a class that normally is a function, the argument "𝐴" is a class
that normally belongs to the domain of the function, and "(𝐹‘𝐴)" is the class that results when the
function is evaluated at the argument. (The result, however, is well-defined for any classes
whatsoever substituted for the class variables, and our definition ends up evaluating to the empty
set when it is not meaningful—see for example Theorem [ndmfv][634].)

Textbooks often use the familiar notation "𝐹(𝐴)" for the value of a function. Outside of context,
this notation is ambiguous: it could also mean the expression that results when 𝐴 is substituted
into the expression that 𝐹 represents. Our left-apostrophe notation, invented by Peano, is often
used by set theorists and has the advantage for us that it is unambiguous and independent of
context.

For special functions such as square root, textbooks indicate function values with an assortment of
two-dimensional glyphs and syntactical idioms that may be ambiguous outside of context. Our versions
may seem unfamiliar because we are constrained to a linear language and we need context-independent,
unambiguous notation. We also prefer a single notation for function value to be able to reuse our
rich collection of theorems about function values. For example, in the square root theorem
[sqrtthi][635], the square root symbol "√" is a function i.e. a class ([csqrt][636]), and we display
the square root of 2 as "(√‘2)". Two other examples are the real part of a complex number *A*, where
we use "(ℜ‘𝐴)" instead of "ℜ𝐴", and the conjugate of a complex number, where we use "(∗‘𝐴)" instead
of "𝐴^{*}" (or sometimes with an overbar), both of which you can see in Theorem [cjval][637].
Another example is the factorial function, which we express as "(!‘𝐴)" instead of "𝐴!" in
[facnn2][638]. In the conventional textbook notation, these four examples use four different
positional relationships to indicate one concept (the value of a function), but I decided to use a
single syntax for all four to make the development of proofs simpler.

There is an exception to the above convention: the unary minus function [df-neg][639] is so common
that I decided to create a special syntax for it. So, the negative of a complex number is
represented as the visually more familiar "-𝐴" rather than the more tedious "(-‘𝐴)". One drawback is
that we have to prove separately certain theorems such as the equality theorem [negeq][640] instead
of being able to reuse the general-purpose [fveq2][641]. Moreover, the bare symbol "-" has no
meaning in isolation, in contrast to say the bare symbol "√", which is itself a set and can be
manipulated as a stand-alone function for various purposes.

Another very important exception is the notation for an operation value, which is the function value
of an ordered pair as defined by [df-ov][642]. It is so commonly used that we adopted the convenient
and familiar notation of three juxtaposed class expressions surrounded by parentheses, such as "( 3
+ 2 )". While this may appear to be syntactically ambiguous with such expressions as the union of
two classes ([cun][643]), "(𝐴 ∪ 𝐵)", the difference is that operation value notation requires that
the center symbol be a class expression: "+" is a class expression ([caddc][644]) but a stand-alone
"∪" is not. (We do not define the union of two classes as an operation because it must work with
proper classes as arguments—not to mention that the operation value definition ultimately depends on
it anyway.)

Prefix expressions (those beginning with a constant, such as the unary minus above, logical
negation, and the "for all" quantifier) have tighter binding than infix expressions and don't
require parentheses. Also, whenever an infix expression is a predicate—i.e. has class variable
arguments but evaluates to a wff, such as A = B—parentheses are not needed for disambiguation. Some
infix predicates that are not surrounded by parentheses are:

A = B
x = y
A e. B
x e. y
A =/= B
A e/ B
A C_ B
A C. B
A R B
R Po A
R Or A
R Fr A
R We A
A Fn B
R _Se A
R _FrSe A
(To find out where say "A Fn B" is defined, type "search * 'A Fn B'" in the metamath program after
reading in set.mm. In this case, df-fn matches, and you can look at the web page [df-fn][645] for
more information.)

There are a several other cases in the notation where infix is used without parentheses; they
include:

F : A --> B
F : A -1-1-> B
F : A -onto-> B
F : A -1-1-onto-> B
H Isom R , S ( A , B )


Appendix 7: Some Predicate Calculus Subsystems

In the set.mm database, there are 10 other axiom schemes of predicate calculus ([ax-c4][646],
[ax-c5][647], [ax-c7][648], [ax-c9][649], [ax-c10][650], [ax-c11][651], [ax-c11n][652],
[ax-c14][653], [ax-c15][654], and [ax-c16][655]) that are not included in our "official" list of 10
[predicate calculus axiom schemes][656] (and one rule) above, because they can be derived as
theorems from the official ones. These used to be part of our official list but were removed or
replaced as redundancies and simplifications were found. We have kept them in a deprecated section
of set.mm for historical purposes, and they are not used outside of that section. Each of these
obsolete axioms is proved as a theorem whose name is the same with the "-" omitted. For example,
obsolete axiom [ax-c5][657] is proved as Theorem [axc5][658] from using Axioms [ax-4][659] through
[ax-13][660].

Axioms with names starting "ax-c" are part of system S3' in Remark 9.6 of [[Megill][661]]; for
example, [ax-c4][662] is axiom C4' there. Axiom [ax-c11n][663] was a newer alternative to
[ax-c11][664], although eventually we proved it redundant (Theorem [axc11n][665]).

From the set consisting of these 10 obsolete axiom schemes plus the 10 official ones, several
subsystems can be of interest for certain studies. For brevity, "axiom" always means "[axiom
scheme][666]" in the table below. In the table, we assume that all 20 axiom schemes and 1 rule (23
axioms and 2 rules if we include propositional calculus) are present, except for the ones listed in
the "Omit axioms" column.


────────┬───────────────────────────────────────────────────────────────────────────────────────────
Omit    │Discussion of resulting subsystem                                                          
axioms  │                                                                                           
────────┼───────────────────────────────────────────────────────────────────────────────────────────
[ax-c5][│As mentioned above, these 10 omitted axioms can be derived from the others (see Theorems   
667],   │[axc5][677], [axc4][678], [axc7][679], [axc10][680], [axc11][681], [axc11n][682],          
[ax-c4][│[axc15][683], [axc9][684], [axc14][685], [axc16][686]). This system is                     
668],   │[scheme-complete][687] and is the one we ordinarily use. It is equivalent to system S3' in 
[ax-c7][│Remark 9.6 of [[Megill][688]].                                                             
669],   │                                                                                           
[ax-c10]│                                                                                           
[670],  │                                                                                           
[ax-c11]│                                                                                           
[671],  │                                                                                           
[ax-c11n│                                                                                           
][672], │                                                                                           
[ax-c15]│                                                                                           
[673],  │                                                                                           
[ax-c9][│                                                                                           
674],   │                                                                                           
[ax-c14]│                                                                                           
[675],  │                                                                                           
and     │                                                                                           
[ax-c16]│                                                                                           
[676]   │                                                                                           
────────┼───────────────────────────────────────────────────────────────────────────────────────────
[ax-5][6│This system is logically complete (see comments in [ax5ALT][690]), but we lose the powerful
89]     │metalogic afforded by the concept of "a variable not occurring in a wff". It is conjectured
        │that [ax-c5][691], [ax-c15][692], and [ax-c14][693] (and possibly other otherwise redundant
        │ones as well) cannot be derived from the others in this system. Proofs in any system       
        │omitting [ax-5][694] tend to be long.                                                      
────────┼───────────────────────────────────────────────────────────────────────────────────────────
Omit all│This is Tarski's system S2 and is logically (though not *scheme*) complete. This means that
predicat│any substitution instance of the omitted axioms that contains no wff metavariables and     
e       │whose setvar variables are mutually distinct can be derived from the axioms of this        
calculus│subsystem. However, many schemes with wff metavariables or bundled setvar variables cannot 
axioms  │be derived. In particular, [ax-10][702], [ax-11][703], [ax-12][704], and [ax-13][705] are  
*except*│conjectured not to be derivable.                                                           
[ax-4][6│                                                                                           
95],    │                                                                                           
[ax-5][6│                                                                                           
96],    │                                                                                           
[ax-6][6│                                                                                           
97],    │                                                                                           
[ax-7][6│                                                                                           
98],    │                                                                                           
[ax-8][6│                                                                                           
99],    │                                                                                           
[ax-9][7│                                                                                           
00], and│                                                                                           
[ax-gen]│                                                                                           
[701]   │                                                                                           
────────┼───────────────────────────────────────────────────────────────────────────────────────────
[ax-c11n│System with no distinct variable conditions. This system has no distinct variable          
][706], │conditions, making the concept of substitution as simple as it can possibly be and also    
[ax-12][│significantly simplifying the algorithm for verifying proofs. It is equivalent to system S2
707],   │in Section 4 of [[Megill][711]]. The primary drawback of this system is that for it to be  
[ax-c15]│considered complete, we must ignore antecedents called "[distinctors][712]" that stand in  
[708],  │for what would be distinct variable conditions (see the linked discussion).                
[ax-c16]│                                                                                           
[709],  │We can optionally include [ax-c15][713] or [ax-12][714] for a somewhat more powerful       
[ax-5][7│metalogic, since these also involve no distinct variable conditions (although without them 
10]     │we can still derive any instance of them not containing wff metavariables).                
        │                                                                                           
        │Proofs in this system tend to be long.                                                     
────────┼───────────────────────────────────────────────────────────────────────────────────────────
[ax-c15]│It is conjectured that [ax-c15][718] cannot be derived from the axioms of this subsystem.  
[715],  │See Item 9b on the [Workshop Miscellany][719] page.                                        
[ax-c16]│                                                                                           
[716],  │                                                                                           
[ax-5][7│                                                                                           
17]     │                                                                                           
────────┼───────────────────────────────────────────────────────────────────────────────────────────
[ax-12][│It is known that [ax-12][723] can be derived from the axioms of this subsystem (see Theorem
720],   │[ax12fromc15][724]).                                                                       
[ax-c16]│                                                                                           
[721],  │                                                                                           
[ax-5][7│                                                                                           
22]     │                                                                                           
────────┼───────────────────────────────────────────────────────────────────────────────────────────
[ax-c11]│It is conjectured that [ax-c11][727] cannot be derived from the axioms of this subsystem.  
[725]   │                                                                                           
and     │                                                                                           
[ax-12][│                                                                                           
726]    │                                                                                           
────────┼───────────────────────────────────────────────────────────────────────────────────────────
Omit all│The subsystem [ax-1][733] through [ax-11][734], [ax-mp][735], and [ax-gen][736], i.e. the  
predicat│axioms not involving equality, is weaker than traditional predicate calculus without       
e       │equality (i.e. that omits the equality axioms labeled stdpc6 and stdpc7 in the table in the
calculus│[traditional predicate calculus][737] section). The reason is that traditional predicate   
axioms  │calculus incorporates proper substitution as part of its metalogic, whereas in our system  
*except*│proper substitution is accomplished directly by the logic itself through our more complex  
[ax-c5][│axioms [ax-7][738] through [ax-5][739]. In our system, substitution and equality are       
728],   │intimately tied in with each other. This "pure" subsystem in effect represents the fragment
[ax-4][7│of logic that remains when equality, proper substitution, and the concept of distinct      
29],    │variables are completely stripped out. Interestingly, if we map "for all" to "necessity"   
[ax-10][│and omit [ax-11][740] from the "pure" subsystem, the result is exactly the modal logic     
730],   │system known as S5 (thanks to Bob Meyer for pointing this out). See also Item 12 on the    
[ax-11][│[Workshop Miscellany][741] page.                                                           
731],   │                                                                                           
and     │Optionally, we can replace [ax-4][742] and [ax-10][743] with [ax-c4][744] and [ax-c7][745] 
[ax-gen]│(provided we replace them both) to obtain an equivalent subsystem. Theorems                
[732]   │[ax4fromc4][746], [ax10fromc7][747], [axc4][748], and [axc7][749] prove the equivalence. In
        │this subsystem, [axc5c711][750] can replace [ax-c5][751], [ax-c7][752], and [ax-11][753].  
────────┴───────────────────────────────────────────────────────────────────────────────────────────


Appendix 8: Axiom Numbering Before December 2018 On 20-Dec-2018, the [predicate calculus][754] axiom
schemes were renumbered to eliminate gaps caused by obsolete or redundant axiom schemes, as well as
to order them according to their current organization. Older papers, slide presentations, and
websites may reference the old numbering, and the table below provides a cross reference.

For further information about this renumbering, see this [Google Group discussion][755] [retrieved
27-Dec-2018] and this [GitHub discussion][756] [retrieved 27-Dec-2018].

Axioms with names starting "ax-c" are now obsolete, but they were part of system S3' in Remark 9.6
of [[Megill][757]]; for example, [ax-c4][758] is axiom C4' there. Axiom [ax-c11n][759] was a newer
alternative to [ax-c11][760], although eventually we proved it redundant (Theorem [axc11n][761]).
Axiom C5 (not primed) is taken from system S3 but is included in S3' as stated in the paper. The
obsolete axioms [ax-c4][762] through [ax-c16][763] are kept for historical reasons in a deprecated
section of set.mm and shouldn't be used outside of that section.

In the "Proof" column, the referenced theorem has two meanings. In the rows for new axiom numbers
[ax-4][764] through [ax-13][765], the "Proof" column theorem shows a proof of the axiom from the
obsolete ones [ax-c4][766] through [ax-c16][767]. This is useful to see how our current axioms can
be derived from the original ones in [[Megill][768]]. In the rows for [ax-c4][769] through
[ax-c16][770], the "Proof" column theorem shows a derivation of the obsolete axiom from our current
system [ax-4][771] through [ax-13][772]. Together, these proofs show that our current system is
equivalent to the one in [[Megill][773]].

──────────┬───────┬────────────┬─────────┬────────────────────────────────┬─────────────────────────
New number│Old    │Proof       │[[Megill]│Content                         │Description              
          │number │            │[774]]   │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-4][775│ax-5​   │[ax4fromc4][│-        │⊢ (∀𝑥(𝜑 → 𝜓) → (∀𝑥𝜑 → ∀𝑥𝜓))     │Quantified Implication   
]         │       │776]        │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-5][777│ax-17​  │-           │C5       │⊢ (𝜑 → ∀𝑥𝜑), where 𝑥 does not   │Distinctness             
]         │       │            │         │occur in 𝜑                      │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-6][778│ax-9​   │[ax6fromc10]│-        │⊢ ¬ ∀𝑥 ¬ 𝑥 = 𝑦                  │Existence                
]         │       │[779]       │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-7][780│ax-8​   │-           │C8'      │⊢ (𝑥 = 𝑦 → (𝑥 = 𝑧 → 𝑦 = 𝑧))     │Equality                 
]         │       │            │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-8][781│ax-13​  │-           │C12'     │⊢ (𝑥 = 𝑦 → (𝑥 ∈ 𝑧 → 𝑦 ∈ 𝑧))     │Left Equality for Binary 
]         │       │            │         │                                │Predicate                
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-9][782│ax-14​  │-           │C13'     │⊢ (𝑥 = 𝑦 → (𝑧 ∈ 𝑥 → 𝑧 ∈ 𝑦))     │Right Equality for Binary
]         │       │            │         │                                │Predicate                
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-10][78│ax-6​   │[ax10fromc7]│-        │⊢ (¬ ∀𝑥𝜑 → ∀𝑥¬ ∀𝑥𝜑)             │Quantified Negation      
3]        │       │[784]       │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-11][78│ax-7​   │-           │C6'      │⊢ (∀𝑥∀𝑦𝜑 → ∀𝑦∀𝑥𝜑)               │Quantifier Commutation   
5]        │       │            │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-12][78│ax-11​  │[ax12fromc15│-        │⊢ (𝑥 = 𝑦 → (∀𝑦𝜑 → ∀𝑥(𝑥 = 𝑦 →    │Substitution             
6]        │       │][787]      │         │𝜑)))                            │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-13][78│ax-12​  │[ax13fromc9]│-        │⊢ (¬ 𝑥 = 𝑦 → (𝑦 = 𝑧 → ∀𝑥 𝑦 = 𝑧))│Quantified Equality      
8]        │       │[789]       │         │                                │                         
──────────┴───────┴────────────┴─────────┴────────────────────────────────┴─────────────────────────
Obsolete axioms                                                                                     
──────────┬───────┬────────────┬─────────┬────────────────────────────────┬─────────────────────────
[ax-c4][79│ax-5o​  │[axc4][791] │C4'      │⊢ (∀𝑥(∀𝑥𝜑 → 𝜓) → (∀𝑥𝜑 → ∀𝑥𝜓))   │Quantified Implication   
0]        │       │            │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-c5][79│ax-4​   │[axc5][793] │C5'      │⊢ (∀𝑥𝜑 → 𝜑)                     │Specialization           
2]        │       │            │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-c7][79│ax-6o​  │[axc7][795] │C7'      │⊢ (¬ ∀𝑥¬ ∀𝑥𝜑 → 𝜑)               │Quantified Negation      
4]        │       │            │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-c9][79│ax-12o​ │[axc9][797] │C9'      │⊢ (¬ ∀𝑧 𝑧 = 𝑥 → (¬ ∀𝑧 𝑧 = 𝑦 → (𝑥│Quantified Equality      
6]        │       │            │         │= 𝑦 → ∀𝑧 𝑥 = 𝑦)))               │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-c10][7│ax-9o​  │[axc10][799]│C10'     │⊢ (∀𝑥(𝑥 = 𝑦 → ∀𝑥𝜑) → 𝜑)         │Existence                
98]       │       │            │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-c11][8│ax-10o​ │[axc11][801]│C11'     │⊢ (∀𝑥 𝑥 = 𝑦 → (∀𝑥𝜑 → ∀𝑦𝜑))      │Quantifier Substitution  
00]       │       │            │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-c11n][│ax-10​  │[axc11n][803│-        │⊢ (∀𝑥 𝑥 = 𝑦 → ∀𝑦 𝑦 = 𝑥)         │Quantifier Substitution  
802]      │       │]           │         │                                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-c14][8│ax-15​  │[axc14][805]│C14'     │⊢ (¬ ∀𝑧 𝑧 = 𝑥 → (¬ ∀𝑧 𝑧 = 𝑦 → (𝑥│Quantified Binary        
04]       │       │            │         │∈ 𝑦 → ∀𝑧 𝑥 ∈ 𝑦)))               │Predicate                
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-c15][8│ax-11o​ │[axc15][807]│C15'     │⊢ (¬ ∀𝑥 𝑥 = 𝑦 → (𝑥 = 𝑦 → (𝜑 →   │Variable Substitution    
06]       │       │            │         │∀𝑥(𝑥 = 𝑦 → 𝜑))))                │                         
──────────┼───────┼────────────┼─────────┼────────────────────────────────┼─────────────────────────
[ax-c16][8│ax-16​  │[axc16][809]│C16'     │⊢ (∀𝑥 𝑥 = 𝑦 → (𝜑 → ∀𝑥𝜑)), where │Distinct Variables       
08]       │       │            │         │𝑥 and 𝑦 are distinct            │                         
──────────┴───────┴────────────┴─────────┴────────────────────────────────┴─────────────────────────


Reading Suggestions Logic and set theory provide a foundation for all of mathematics. One possible
way to start to learn about them is to use the Metamath Proof Explorer in conjunction with one or
more textbooks listed in the Bibliography of the next section. The textbooks provide a motivation
for what we are doing, whereas the Metamath Proof Explorer lets you see in detail all hidden and
implicit steps. Most standard theorems are accompanied by [literature cross references][810], and it
will probably be easier to acquire a higher-level understanding of them if you consult these.

For propositional and predicate calculus, Margaris is now available in an inexpensive Dover edition
and is reasonably readable for beginners, once you get used to the archaic dot notation used in
place of parentheses. Our 19.x series of theorems, such as [19.35][811], corresponds to Margaris'
handy predicate calculus table on pp. 89-90. Hirst and Hirst's on-line *A Primer for Logic and
Proof*, mentioned [above][812], uses modern notation and is also highly recommended.

For set theory, Quine is wonderfully written and a pleasure to read. The first part on virtual
classes (pp. 15-21) is a must-read if you want to understand our purple class variables (which in
Quine are Greek letters). (See also the [Theory of Classes][813] above.) Quine also uses the archaic
dot notation, but this is really a very minor hurdle, especially since you can compare the Metamath
versions of many of the theorems. Takeuti and Zaring is the set theory reference we follow most
closely, including most of our notation, and its rigor makes it straightforward to formalize proofs;
but some people find it a dry and technical read, and it is also out of print. Suppes, which is
available in a Dover edition, goes to extremes to *avoid* virtual classes, leading to bizarre
theorems like "the set of all sets is empty" (Theorem 50, p. 35); nonetheless, it provides a gentle
introduction that we reference surprisingly frequently.

Margaris and Quine give only an informal explanation of their dot notation that may not satisfy
everyone. A good explanation can be found in Copi's *Symbolic Logic* (5th edition), Section 9.3,
with multiple examples. (Thanks to Patrick Clot for this suggestion.)

Some closely followed texts in the Metamath Proof Explorer are listed below. I am not specifically
endorsing them but just indicating which books and papers I consulted frequently while building the
database.

* Axioms of propositional calculus—Margaris.
* Theorems of propositional calculus—Whitehead and Russell.
* Axioms of predicate calculus—Megill (System S3' in the article referenced).
* Theorems of pure predicate calculus—Margaris.
* Theorems of equality and substitution—Monk2, Tarski, Megill.
* Axioms of set theory—Bell and Machover.
* Virtual classes in set theory (our class builder notation and our purple class variables)—Quine.
* Development of set theory (including most of the notation we use)—Takeuti and Zaring.
* Construction of real and complex numbers—Gleason.
* Elementary theorems about real numbers—Apostol.
Bibliography This is the complete list of books and articles that are [cross referenced][814] in the
Metamath Proof Explorer database. The numbers in brackets are the Library of Congress
classifications to make them easier to find on your university library shelves. These may differ
slightly for resources assigned by the libary itself, and the ones we show are used by the MIT
libraries.

1.   [Adamek] Jiří Adámek, Horst Herrlich, George E. Strecker, *Abstract and Concrete Categories:
     The Joy of Cats*, Dover Publications, Mineola, New York (2004); available at [
     http://katmat.math.uni-bremen.de/acc][815] (retrieved 2-Jan-2017).
2.   [AhoHopUll] Alfred V. Aho, John E. Hopcroft and Jeffrey D. Ullman, *The Design and Analysis of
     Computer Algorithms*, Addison-Wesley, Reading, Massachusetts (1974) [QA76.6 .A36].
3.   [AitkenIBG] Aitken, Wayne, "Incidence Betweenness Geometry," Handout (2008); available at
     [http://public.csusm.edu/aitken_html/m410/betweenness.08.pdf][816] (retrieved 6 Apr 2016). [The
     series of handouts is available at [http://public.csusm.edu/aitken_html/m410/][817] (retrieved
     6 Apr 2016).]
4.   [AitkenIBCG] Aitken, Wayne, "IBC Geometry," Handout (2008); available at
     [http://public.csusm.edu/aitken_html/m410/congruence.08.pdf][818] (retrieved 6 Apr 2016).
5.   [AitkenNG] Aitken, Wayne, "Neutral Geometry," Handout (2010); available at
     [http://public.csusm.edu/aitken_html/m410/neutral.10.pdf][819] (retrieved 6 Apr 2016).
6.   [AitkenEG] Aitken, Wayne, "Euclidean Geometry," Handout (2010); available at
     [http://public.csusm.edu/aitken_html/m410/euclidean.10.pdf][820] (retrieved 6 Apr 2016).
7.   [AitkenQ] Aitken, Wayne, "Quadrilaterals," Handout (2008); available at
     [http://public.csusm.edu/aitken_html/m410/quad.08.pdf][821] (retrieved 6 Apr 2016).
8.   [AitkenLDZT] Aitken, Wayne, "Legendre's Defect Zero Theorem," Handout (2008); available at
     [http://public.csusm.edu/aitken_html/m410/defectzero.08.pdf][822] (retrieved 6 Apr 2016).
9.   [AitkenEHC] Aitken, Wayne, "Euclidean and Hyperbolic Conditions," Handout (2008); available at
     [http://public.csusm.edu/aitken_html/m410/conditions.08.pdf][823] (retrieved 6 Apr 2016).
10.  [AitkenTHG] Aitken, Wayne, "Topics in Hyperbolic Geometry," Handout (2008); available at
     [http://public.csusm.edu/aitken_html/m410/hyperbolic.08.pdf][824] (retrieved 6 Apr 2016).
11.  [AkhiezerGlazman] Akhiezer, N. I., and I. M. Glazman, *Theory of Linear Operators in Hilbert
     Space*, Vol. I, Dover Publications, Mineola, New York (1993) [QA322.4.A3813 1993].
12.  [Alling] Alling, Norman L., *Foundations of Analysis Over Surreal Number Fields,* Elsevier
     Science Publishing Company, Inc. (1987) [QA1.N79].
13.  [Apostol] Apostol, Tom M., *Calculus,* vol. 1, 2nd ed., John Wiley & Sons Inc. (1967)
     [QA300.A645 1967].
14.  [ApostolNT] Apostol, Tom M., *Introduction to analytic number theory,* Springer-Verlag, New
     York, Heidelberg, Berlin (1976) [QA241.A6].
15.  [Baer] Baer, Reinhold, *Linear Algebra and Projective Geometry,* Academic Press, New York
     (1952) [QA471.B34].
16.  [Bauer] Bauer, Andrej, "Five stages of accepting constructive mathematics," *Bulletin (New
     Series) of the American Mathematical Society*, 54:481-498 (2017), DOI: [10.1090/bull/1556][825]
     .
17.  [BeauregardFraleigh] Beauregard, Raymond A., and Fraleigh, John B., *A First Course In Linear
     Algebra: with Optional Introduction to Groups, Rings, and Fields,* Houghton Mifflin Company,
     Boston (1973).
18.  [BellMachover] Bell, J. L., and M. Machover, *A Course in Mathematical Logic,* North-Holland,
     Amsterdam (1977) [QA9.B3953].
19.  [Beeson2016] Beeson, Michael, and Larry Wos, "Finding Proofs in Tarskian Geometry", 2016-06-22,
     *Journal of Automated Reasoning*, Volume 58, DOI 10.1007/s10817-016-9392-2, [
     ResearchGate][826] or [ Prepublication][827].
20.  [BeltramettiCassinelli1] Enrico G. Beltrametti and Gianni Cassinelli, "Logical and Mathematical
     Structures of Quantum Mechanics," *La Rivista del Nuovo cimento* 6:321-404 (1976) [QC.R625].
21.  [BeltramettiCassinelli] Enrico G. Beltrametti and Gianni Cassinelli, *The Logic of Quantum
     Mechanics* (Encyclopedia of Mathematics and Its Applications; v. 15, Mathematics of Physics),
     Addison-Wesley, Reading, Massachusetts (1981) [QC174.12.B45].
22.  [Beran] Ladislav Beran, *Orthomodular Lattices; Algebraic Approach*, D. Reidel, Dordrecht
     (1985) [QA171.5.B4].
23.  [Berge] Berge, Claude *Hypergraphs*, Elsevier Science B.V., Amsterdam (1989) [QA166.23.B4813
     1989]
24.  [Bobzien] Bobzien, Susanne, "Stoic Logic", *The Cambridge Companion to Stoic Philosophy*, Brad
     Inwood (ed.), Cambridge University Press (2003-2006), [ https://philpapers.org/rec/BOBSL][828].
25.  [Bogachev] Bogachev, V.I., *Measure Theory, Volume I* Springer-Verlag, Berlin, New York, (2007)
     [QC20.7.M34.B64 2007].
26.  [Bollobas] Béla Bollobás, *Modern Graph Theory*, Graduate Texts in Mathematics, Springer
     Science+Media New York (1998) [QA166.B663].
27.  [BourbakiEns] Bourbaki, Nicolas, *Théorie des ensembles,* Springer-Verlag, Berlin Heidelberg
     (2007); available in English (for purchase) at [
     http://www.springer.com/us/book/9783540225256][829] (retrieved 15-Aug-2016). Page references in
     set.mm are for the French edition.
28.  [BourbakiAlg1] Bourbaki, Nicolas, *Algèbre, Chapitres 1 à 3,* Springer-Verlag, Berlin
     Heidelberg (2007); available in English (for purchase) at [
     http://www.springer.com/us/book/9783540642435][830] (retrieved 15-Aug-2016). Page references in
     set.mm are for the French edition.
29.  [BourbakiAlg2] Bourbaki, Nicolas, *Algèbre, Chapitres 4 à 7,* Springer-Verlag, Berlin
     Heidelberg (2007); available in English (for purchase) at [
     https://link.springer.com/book/10.1007/978-3-642-61698-3][831] (retrieved 3-Aug-2025). Page
     references in set.mm are for the French edition.
30.  [BourbakiCAlg2] Bourbaki, Nicolas, *Algèbre Commutative, Chapitres 5 à 7,* Springer-Verlag,
     Berlin Heidelberg (2006); available in English (for purchase) at [
     https://link.springer.com/book/9783540642398][832] (retrieved 15-Jun-2025). Page references in
     set.mm are for the French edition.
31.  [BourbakiTop1] Bourbaki, Nicolas, *Topologie générale, Chapitres 1 à 4,* Springer-Verlag,
     Berlin Heidelberg (2007); available in English (for purchase) at [
     http://www.springer.com/us/book/9783540642411][833] (retrieved 15-Aug-2016). Page references in
     set.mm are for the French edition.
32.  [BourbakiTop2] Bourbaki, Nicolas, *Topologie générale, Chapitres 5 à 10,* Springer-Verlag,
     Berlin Heidelberg (2007); available in English (for purchase) at [
     http://www.springer.com/us/book/9783540645634][834] (retrieved 15-Aug-2016). Page references in
     set.mm are for the French edition.
33.  [BourbakiFVR] Bourbaki, Nicolas, *Fonctions d'une variable réelle,* Springer-Verlag, Berlin
     Heidelberg (2007); available in English (for purchase) at [
     http://www.springer.com/us/book/9783540653400][835] (retrieved 15-Aug-2016). Page references in
     set.mm are for the French edition.
34.  [BrosowskiDeutsh] Bruno Brosowsky and Frank Deutsh, "An elementary proof of the
     Stone-Weierstrass theorem", Proc. Amer. Math. Soc. 81 (1981), 89-92; available at [
     http://www.ams.org/journals/proc/1981-081-01/S0002-9939-1981-0589143-8/][836] (retrieved
     24-Apr-2017).
35.  [Bruck] Bruck, Richard Hubert *A survey of binary systems*, 3rd ed., Springer-Verlag Berlin
     Heidelberg New York (1971)
36.  [ChoquetDD] Choquet-Bruhat, Yvonne and Cecile DeWitt-Morette, with Margaret Dillard-Bleick,
     *Analysis, Manifolds and Physics,* Elsevier Science B.V., Amsterdam (1982) [QC20.7.A5C48 1981].
37.  [Church] Church, Alonzo, *Introduction to Mathematical Logic,* Princeton University Press,
     1956.
38.  [Clemente] Clemente Laboreo, Daniel *Introduction to natural deduction* (2014); available at [
     http://www.danielclemente.com/logica/dn.en.pdf][837] (PDF) and [
     http://www.danielclemente.com/logica/dn.en.html][838] (HTML) (retrieved 10 Feb 2017).
39.  [Cohen] Cohen, David, *Precalculus With Unit-Circle Trigonometry,* Brooks-Cole, Pacific Grove,
     CA (1988) [QA331.3.C64].
40.  [Cohen4] Cohen, David, *Precalculus With Unit-Circle Trigonometry*, 4th ed., Brooks-Cole,
     Pacific Grove, CA (2006) [QA331.3.C64].
41.  [Cohn] Cohn, Paul Moritz, *Universal Algebra,* D. Reidel, Dordrecht (1981) [QA251.C55].
42.  [CohnMT] Cohn, Donald L., *Measure Theory,* Birkhäuser, New York (2013).
43.  [Connell] Edwin H. Connell, *Elements of Abstract and Linear Algebra,* Orange Grove Texts Plus
     (October 21, 2009); ISBN 978-1616100032; available at [
     http://www.math.miami.edu/~ec/book][839] (retrieved 10 Dec 2019).
44.  [Conway] John H. Conway, *On Numbers and Games,* 2nd ed., A K Peters Ltd (2001) [QA241.C69].
45.  [CormenLeisersonRivest] Cormen, Thomas, Charles E. Leiserson, and Ronald L. Rivest,
     *Introduction to Algorithms,* The MIT Press, Cambridge, Massachusetts (1989) [QA76.6.C662].
46.  [Crawley] Crawley, Peter and Robert P. Dilworth, *Algebraic Theory of Lattices,* Prentice-Hall,
     Englewood Cliffs, New Jersey (1973) [QA171.5.C7].
47.  [Diestel] Diestel, Reinhard, *Graph Theory, 5th Electronic Edition,* Springer-Verlag,
     Heidelberg (2016) [QA166.D51413]. URL: [ http://diestel-graph-theory.com/index.html][840]
48.  [Eisenberg] Eisenberg, Murray, *Axiomatic Theory of Sets and Classes,* Holt, Rinehart and
     Winston, Inc., New York (1971) [QA248.E36].
49.  [Enderton] Enderton, Herbert B., *Elements of Set Theory,* Academic Press, Inc., San Diego,
     California (1977) [QA248.E5].
50.  [Euclid1] Euclid, *The Thirteen Books of Euclid's Elements,* Vol. I: Introduction and books
     I—II, translated by Thomas L. Heath, Cambridge University Press, Cambridge (1908); available at
     [ https://archive.org/details/thirteenbookseu03heibgoog][841] (retrieved 9 Apr 2016). See also
     David Joyce's web pages at [ http://aleph0.clarku.edu/~djoyce/java/elements/][842] for
     interactive Java applets and detailed expositions.
51.  [Euclid2] Euclid, *The Thirteen Books of Euclid's Elements,* Vol. II: Books III—IX, translated
     by Thomas L. Heath, Cambridge University Press, Cambridge (1908); available at [
     https://archive.org/details/thirteenbookseu03heibgoog][843] (retrieved 9 Apr 2016).
52.  [Euclid3] Euclid, *The Thirteen Books of Euclid's Elements,* Vol. III: Books X—XIII and
     appendix, translated by Thomas L. Heath, Cambridge University Press, Cambridge (1908);
     available at [ https://archive.org/details/thirteenbookseu03heibgoog][844] (retrieved 9 Apr
     2016).
53.  [FaureFrolicher] Faure, Claude-Alain and Frölicher, Alfred, *Modern Projective Geometry,*
     Springer, Dordrecht (2000) [QA471.F33 2000].
54.  [Frege1879] Frege, Gottlob, *Begriffsschrift* (German for "concept-script"), [
     https://gallica.bnf.fr/ark:/12148/bpt6k65658c][845].
55.  [Fremlin1] Fremlin, D. H., *Measure Theory, Vol. 1: The Irreducible Minimum*, 2nd ed., Lulu
     Press, Morrisville, North Carolina (2011) [QA312.F72]; available at
     [https://wiki.math.ntnu.no/_media/tma4225/2011/fremlin-mt1.pdf][846] (retrieved 14 Apr 2015).
56.  [Fremlin5] Fremlin, D. H., *Measure Theory, Vol. 5: Set-theoretic Measure Theory* Lulu Press,
     Morrisville, North Carolina (2008) [QA312.F72]; available at
     [https://www.essex.ac.uk/maths/people/fremlin/mt.htm][847] (retrieved 14 Apr 2015).
57.  [FreydScedrov] Freyd, Peter J. and Andre Scedrov, *Categories, Allegories,* Elsevier Science
     Publishers B.V., Amsterdam (1990) [QA169.F73 1990].
58.  [Gleason] Gleason, Andrew M., *Fundamentals of Abstract Analysis,* Jones and Bartlett
     Publishers, Boston (1991) [QA300.G554].
59.  [Golan] Golan, Jonathan S., *Semirings and their Applications,* Kluwer Academic Publishers,
     Dordrecht (1999) [QA251.5 .G642 1999].
60.  [Gonshor] Gonshor, Harry, *An Introduction to the Theory of Surreal Numbers,* London
     Mathematical Society (1986) [QA241.G63 1986].
61.  [Gratzer] Grätzer, George, *Universal Algebra,* 2nd ed. with updates, Springer, New York (2008)
     [QA251.G68 2008].
62.  [GramKnuthPat] Gram, Ronald L., Knuth, Donald E., and Patashnik, Oren, *Concrete Mathematics,*
     2nd Ed., Addison-Wesley Publishing Company, Reading, MA (1994) [QA39.2.G733].
63.  [EGA] Grothendieck, Alexandre and Dieudonné, Jean *Éléments de géométrie algébrique* Institut
     des hautes études scientifiques, Publ. Math. No. 4 (1960), available at [ NumDam][848].
64.  [Hall] Hall, Marshall Jr., *The Theory of Groups,* The Macmillan Company, New York (1959)
     [QA171.H27 2018].
65.  [Halmos] Paul R. Halmos, *Introduction to Hilbert Space and the Theory of Spectral
     Multiplicity*, AMS Chelsea Publishing, Providence, Rhode Island (1957) [QA691.H194 1957].
66.  [Hamilton] Hamilton, A. G., *Logic for Mathematicians,* Cambridge University Press, Cambridge,
     revised edition (1988) [QA9.H298 1988].
67.  [Hatcher] Hatcher, Allen, *Algebraic Topology,* Cambridge University Press, Cambridge (2002)
     [QA612.H42 2002]; available at [ http://www.math.cornell.edu/~hatcher/AT/ATpage.html][849]
     (retrieved 11 Nov 2014).
68.  [Hefferon] Hefferon, Jim, *Linear Algebra*, 3rd edition, Lightning Source Inc. (2017), ISBN-13:
     978-1944325039, available at [ http://joshua.smcvt.edu/linearalgebra][850] (retrieved 26 Nov
     2019).
69.  [Helfgott] Helfgott, Harald A., *The ternary Goldbach conjecture is true* arXiv:1312.7748v2
     (17-Jan-2014); available at [ https://arxiv.org/abs/1312.7748][851] (retrieved 2 Aug 2020). See
     also [ https://webusers.imj-prg.fr/~harald.helfgott/anglais/book.html][852]
70.  [Herstein] Herstein, I. N., *Abstract Algebra,* Macmillan Publishing Company, New York (1986)
     [QA162.H47 1986].
71.  [Hitchcock] Hitchcock, David, *The peculiarities of Stoic propositional logic*, McMaster
     University; available at [ http://www.humanities.mcmaster.ca/~hitchckd/peculiarities.pdf][853]
     (retrieved 3 Jul 2016).
72.  [Holland95] Holland, Samuel S., "Orthomodularity in infinite dimensions; a theorem of M.
     Solèr," *Bull. Am. Math. Soc.* 32:205-234 (1995) [QA.A513]; available at [
     http://arxiv.org/abs/math/9504224v1][854] (retrieved 11 Nov 2014).
73.  [Holmes] Holmes, Robert, *Elementary Set Theory With a Universal Set*; available at [
     https://randall-holmes.github.io/head.pdf][855] (retrieved 7 Feb 2022).
74.  [Huneke] Huneke, Craig L., "The Friendship Theorem" *American Mathematical Monthly* 109:192-194
     (2002); available at [
     https://www.researchgate.net/publication/268313555_The_Friendship_Theorem][856] (retrieved 30
     Dec 2017).
75.  [Indrzejczak] Indrzejczak, Andrzej, *Natural Deduction, Hybrid Systems and Modal Logic*, Trends
     in Logic Vol. 30, Springer (2010).
76.  [Jech] Jech, Thomas, *Set Theory,* Academic Press, San Diego (1978) [QA248.J42].
77.  [JonesMatijasevic] Jones, J. P. and Y. V. Matijasevič (Matiyasevich), "Proof of Recursive
     Unsolvability of Hilbert's Tenth Problem," *American Mathematical Monthly,* 98:689-709 (1991)
     [QA.A5125]; available at
     [https://www.williamstein.org/edu/Spring2003/21n/papers/hilbert10.pdf][857] (retrieved 11 Nov
     2014).
78.  [Juillerat] Jacob Juillerat, "Transcendental Numbers", Lake Forest College, Senior Thesis,
     April 25, 2016 ; available at [ https://publications.lakeforest.edu/seniortheses/62/][858]
     (retrieved 5-Apr-2020).
79.  [JustWeese] Just, Winfried, and Martin Weese, *Discovering Modern Set Theory I: The Basics,*
     The American Mathematical Society, Providence, R.I. (1995) [QA248.J87].
80.  [KalishMontague] Kalish, D. and R. Montague, "On Tarski's formalization of predicate logic with
     identity," *Archiv für Mathematische Logik und Grundlagenforschung,* 7:81-101 (1965) [QA.A673];
     available at [http://gdz.sub.uni-goettingen.de/dms/load/img/?PID=GDZPPN002043211][859]
     (retrieved 29-Apr-2017).
81.  [Kalmbach] Kalmbach, Gudrun *Orthomodular Lattices*, Academic Press, London (1983)
     [QA171.5.K34].
82.  [KanamoriPincus] Kanamori, A. and D. Pincus, "Does GCH Imply AC Locally?", in Gábor Halász, et
     al.(eds.), *Paul Erdős and his Mathematics,* *Bolyai Society Mathematical Studies* 11:413-426,
     János Bolyai Mathematical Society, Budapest (2002) [QA1.P194 2002]; available at
     [http://math.bu.edu/people/aki/7.pdf][860] (retrieved 4 Jun 2015).
83.  [Kreyszig] Kreysig, Erwin, *Introductory Functional Analysis with Applications*, John Wiley &
     Sons, New York (1989) [QA320.K74].
84.  [Kulpa] Kulpa, Wladislaw, "The Poincaré- Miranda Theorem", *American Mathematical Monthly*
     104:545-550 (1997); available at [ https://www.jstor.org/stable/2975081][861] (retrieved 6 Feb
     2019).
85.  [Kunen] Kunen, Kenneth, *Set Theory: An Introduction to Independence Proofs,* Elsevier Science
     B.V., Amsterdam (1980) [QA248.K75].
86.  [Kunen2] Kunen, Kenneth, *Set Theory,* revised edition, College Publications, London (2013)
     [QA248 .K86 2013].
87.  [KuratowskiMostowski] Kuratowski, K. and A. Mostowski, *Set Theory: with an Introduction to
     Descriptive Set Theory,* 2nd ed., North-Holland, Amsterdam (1976) [QA248.K7683 1976].
88.  [Lang] Serge Lang, *Algebra*, revised 3rd edition, Graduate Texts in Mathematics,
     Springer-Verlag New York (2002) [QA154.3.L3].
89.  [LarsonHostetlerEdwards] Larson, Ron, Robert P. Hostetler, and Bruce H. Edwards, *Calculus:
     Early Transcendental Functions,* 3rd ed., Houghton Mifflin Company (2003) [QA303.L3274 2003]
90.  [LeBlanc] LeBlanc, Hugues, "On Meyer and Lambert's Quantificational Calculus FQ," *J. Symb.
     Logic* 33:275-280 (1968) [QA.J87].
91.  [Levy58] Lévy, A., "The independence of various definitions of finiteness," *Fundamenta
     Mathematicae*, 46:1-13 (1958) [QA.F981]
92.  [Levy] Levy, Azriel, *Basic Set Theory*, Dover Publications, Mineola, N.Y. (2002) [QA248.L398
     2002].
93.  [Lipparini] Lipparini, Paolo, *A clean way to separate sets of surreals* arXiv:1712.03500
     (20-May-2018); available at [ https://arxiv.org/abs/1712.03500][862] (retrieved 7 Dec 2021).
94.  [Lopez-Astorga] Lopez-Astorga, Miguel, "The First Rule of Stoic Logic and its Relationship with
     the Indemonstrables", *Revista de Filosofía Tópicos* (2016); available at [
     http://www.scielo.org.mx/pdf/trf/n50/n50a1.pdf][863] (retrieved 3 Jul 2016).
95.  [MaedaMaeda] Maeda, Fumitomo (1897-1965) and Shuichiro Maeda, *Theory of Symmetric Lattices*,
     Springer-Verlag, New York (1970) [Q171.5.M184].
96.  [Margaris] Margaris, Angelo, *First Order Mathematical Logic,* Blaisdell Publishing Company,
     Waltham, Massachusetts (1967) [QA9.M327].
97.  [McKMegPav] McKay, B., N. Megill, and M. Pavicic, "Algorithms for Greechie Diagrams," *Int. J.
     Theor. Phys.,* 39:2393-2417 (2000) [QC.I626]; available at [
     http://xxx.lanl.gov/abs/quant-ph/0009039][864] (retrieved 11 Nov 2014).
98.  [Megill] Megill, N., "A Finitely Axiomatized Formalization of Predicate Calculus with
     Equality," *Notre Dame Journal of Formal Logic,* 36:435-453 (1995) [QA.N914]; available at
     [http://projecteuclid.org/euclid.ndjfl/1040149359][865] (retrieved 11 Nov 2014); the [PDF
     preprint][866] has the same content (with corrections) but pages are numbered 1-22, and the
     database references use the numbers printed on the page itself, not the PDF page numbers. See
     [technical note 1][867] above for some additional notes on this paper.
99.  [MegPav2000] Megill, N. and M. Pavičić, "Equations, States, and Lattices of
     Infinite-Dimensional Hilbert Space," *Int. J. Theor. Phys.* 39:2337-2379 (2000) [QC.I626];
     available at [http://xxx.lanl.gov/abs/quant-ph/0009038][868] (retrieved 11 Nov 2014).
100. [MegPav2002] Megill, N. and M. Pavičić "Deduction, Ordering, and Operations in Quantum Logic,"
     *Found. Phys.* 32:357-378 (2002) QC.F771; available at
     [http://xxx.lanl.gov/abs/quant-ph/0108074][869] (retrieved 11 Nov 2014).
101. [Mendelson] Mendelson, Elliott, *Introduction to Mathematical Logic,* 2nd ed., D. Van Nostrand
     (1979) [QA9.M537].
102. [MertziosUnger] George B. Mertzios, Walter Unger, *The friendship problem on graphs*,
     ROGICS'08, 12-17 May 2008, Mahida, Tunisia, pp. 152-158, or Journal of Multiple-Valued Logic
     and Soft Computing, Vol. 27(2-3), 2016, pp. 275-285 (Page references in set.mm are for the
     ROGICS'08 paper); available at [
     http://community.dur.ac.uk/george.mertzios/papers/Conf/Conf_Windmills.pdf][870] (retrieved
     28-Nov-2017).
103. [Monk1] Monk, J. Donald, *Introduction to Set Theory,* McGraw-Hill, Inc. (1969) [QA248.M745].
104. [Monk2] Monk, J. Donald, "Substitutionless Predicate Logic with Identity," *Archiv für
     Mathematische Logik und Grundlagenforschung,* 7:103-121 (1965) [QA.A673]; available at
     [http://euclid.colorado.edu/~monkd/monk07.pdf][871] (retrieved 29-Apr-2017) or
     [http://gdz.sub.uni-goettingen.de/dms/load/img/?PID=GDZPPN00204322X][872] (retrieved
     29-Apr-2017).
105. [Moore] Moore, Eliakim Hastings, *Introduction to a Form of General Analysis,* Yale University
     Press, New Haven (1910) [QA331.M78].
106. [Moschovakis] Moschovakis, Yiannis N., *Notes on Set Theory,* Springer, New York, second
     edition (2006) [QA248.M665 2006].
107. [Munkres] Munkres, James Raymond, *Topology: a first course,* Prentice-Hall, Englewood Cliffs,
     New Jersey (1975) [QA611.M82].
109. [Nathanson] Nathanson, Melvyn B., *Additive Number Theory: The Classical Bases* Springer-Verlag
     (1996) [QA241.N347 1996].
110. [OeSilva] Tomás Oliveira e Silva, Siegfried Herzog, and Silvio Pardi, *Empirical verification
     of the even Goldbach conjecture and computation of prime gaps up to 4 x 10^18,* Mathematics of
     Computation, vol. 83, no. 288, pp. 2033-2060, July 2014 (published electronically on November
     18, 2013); available at [
     https://www.ams.org/journals/mcom/2014-83-288/S0025-5718-2013-02787-1/home.html][873].
111. [Pfenning] Pfenning, Frank, *Automated Theorem Proving,* Carnegie-Mellon University (April 13,
     2004); available at [ http://www.cs.cmu.edu/~fp/courses/atp/handouts/atp.pdf][874] and [
     http://web.archive.org/web/20160304013704/http://www.cs.cmu.edu/~fp/courses/atp/handouts/atp.pd
     f][875] (retrieved 7 Feb 2017).
112. [Peano] Peano, Giuseppe. *Arithmetices principia: Nove Methodo Ezposita*, 1889.
113. [Ponnusamy] Ponnusamy, S., *Foundations of Functional Analysis,* Alpha Science International
     Ltd., Pangbourne (2002) [QA320.P66 2002].
114. [PtakPulmannova] Pavel Pták and Sylvia Pulmannová, *Orthomodular Structures as Quantum Logics*,
     Kluwer Academic Publishers, Dordrecht (1991) [QC174.17.M35P7713].
115. [Quine] Quine, Willard van Orman, *Set Theory and Its Logic,* Harvard University Press,
     Cambridge, Massachusetts, revised edition (1969) [QA248.Q7 1969].
116. [ReedSimon] Michael Reed and Barry Simon, *Methods of Modern Mathematical Physics*, Vol. I:
     Functional Analysis, Academic Press, New York (1972) [QC20.7.F84.R43].
117. [Roman] Steven Roman, *Advanced Linear Algebra*, 3rd edition, Graduate Texts in Mathematics,
     Springer Science+Business Media, New York (2008) [QA184.R65 1992].
118. [Rosenlicht] Maxwell Rosenlicht, *Introduction to Analysis,* Dover Publications, Inc., New York
     (1986) [QA300.R63].
119. [Rosser] Rosser, John B., *Logic for Mathematicians*, Dover Publications, Mineola, N.Y. (2008)
     [BC135.R58 2008].
120. [Rosser2] Rosser, J. Barkley and Schoenfeld, Lowell, *Approximate formulas for some functions
     of prime numbers*, Illinois J. Math. 6:64-94 (1962); available at [Project Euclid][876]
     (retrieved 27 Dec 2021).
121. [Rotman] Rotman, Joseph J., *The theory of groups,* Allyn and Bacon, Inc., Boston (1973).
122. [Rudin] Rudin, Walter, *Principles of Mathematical Analysis,* McGraw-Hill, New York, second
     edition (1964) [QA300.R916 1964].
123. [Sanford] Sanford, David H., *If P, then Q: Conditionals and the Foundations of Reasoning*, 2nd
     ed., Routledge Taylor & Francis Group (2003); ISBN 0-415-28369-8; available at
     [https://books.google.com/books?id=h_AUynB6PA8C&pg=PA39#v=onepage&q&f=false][877] (retrieved 3
     Jul 2016).
124. [Schechter] Schechter, Eric, *Handbook of Analysis and Its Foundations*, Academic Press, San
     Diego (1997) [QA300.S339].
125. [Schloeder] Schlöder, Julian J., *Ordinal Arithmetic*, Tutorial notes on Set Theory, Bonn
     (2012); available at [https://jjsch.github.io/output/oa.pdf][878] (retrieved 5 Feb 2025).
126. [Schwabhauser] Schwabhauser, Wolfram, Wanda Szmielew, and Alfred Tarski, *Metamathematische
     Methoden in der Geometrie*, Springer-Verlag, Berlin, New York (1983) [QA481.S38]
127. [Shapiro] Shapiro, Harold N., *Introduction to the Theory of Numbers,* Dover Publications, Inc.
     (2008) [QA241.S445 1983].
128. [Stewart] Stewart, Ian Nicholas, *Galois Theory*, 4th ed., CRC Press (2015) [QA214.S74 2015];
     ISBN 978-1-4822-4583-7; available at [
     https://eclass.uoa.gr/modules/document/file.php/MATH594/Stewart%20Galois%204th%20edition.pdf/][
     879] (retrieved 15 Nov 2025).
129. [Stoll] Stoll, Robert R., *Set Theory and Logic,* Dover Publications, Inc. (1979) [QA248.S7985
     1979].
130. [Strang] Strang, Gilbert, *Calculus*, 1st ed., Wellesley-Cambridge Press (1991) [QA303.S8839
     1991]; ISBN 978-09802327-45; available at [
     http://ocw.mit.edu/resources/res-18-001-calculus-online-textbook-spring-2005/][880] (retrieved
     24 Nov 2015).
131. [Sullivan2004] Sullivan, Peter M, "Frege's Logic", *Handbook of the History of Logic, Volume 3:
     The Rise of Modern Logic from Leibniz to Frege*, Edited by Dov M. Gabbay and John WOods, 2004,
     Elsevier, ISBN 0-444-51611-5.
132. [Suppes] Suppes, Patrick, *Axiomatic Set Theory,* Dover Publications, Inc. (1972) [QA248.S959].
133. [Szendrei] Szendrei, Ágnes, *Clones in Universal Algebra*, Les Presses de l'Université de
     Montréal, Montreal, Quebec (1986).
134. [Siegrist] Siegrist, Kyle &al *Random: Probability, Mathematical Statistics, Stochastic
     Processes* available at [ http://www.randomservices.org/random/index.html][881] (retrieved Oct
     2017).
135. [TakeutiZaring] Takeuti, Gaisi, and Wilson M. Zaring, *Introduction to Axiomatic Set Theory,*
     Springer-Verlag, New York, second edition (1982) [QA248.T136 1982].
136. [Tarski] Tarski, Alfred, "A Simplified Formalization of Predicate Logic with Identity," *Archiv
     für Mathematische Logik und Grundlagenforschung,* 7:61-79 (1965) [QA.A673]; available at
     [http://gdz.sub.uni-goettingen.de/dms/load/img/?PID=GDZPPN002043203][882] (retrieved
     29-Apr-2017).
137. [Tarski1999] Tarski, Alfred, and Steven Givant, "Tarski's System of Geometry" *The Bulletin of
     Symbolic Logic*, Volume 5, Number 2, June 1999, pp. 175-214.
138. [Truss] Truss, John Kenneth, *Foundations of Mathematical Analysis,* Oxford University Press,
     Oxford (1997) [QA299.8.T78 1997].
139. [vandenDries] van den Dries, Lou, "Recursion Theory Notes, Fall 2011," available at
     [http://www.math.uiuc.edu/~vddries/recursion.pdf][883] (retrieved 11 Nov 2014).
140. [Viaclovsky7] Viaclovsky, Jeff, *Measurability and Integration, Fall 2003, Lecture 7,*
     available at [
     https://ocw.mit.edu/courses/mathematics/18-125-measure-and-integration-fall-2003/lecture-notes/
     18125_lec7.pdf ][884] (retrieved 23-Mar-2018).
141. [Viaclovsky8] Viaclovsky, Jeff, *Measurability and Integration, Fall 2003, Lecture 8,*
     available at [
     https://ocw.mit.edu/courses/mathematics/18-125-measure-and-integration-fall-2003/lecture-notes/
     18125_lec8.pdf ][885] (retrieved 21-Feb-2018).
142. [Weierstrass] Weierstraß, Karl, *Zur Determinantentheorie,* 1886/87, posthumously published
     1903 in "Mathematische Werke", vol. III, 271-287, J. Knoblauch, Ed. Berlin: Mayer & Müller;
     available at
     [https://www.yumpu.com/de/document/read/20688945/mathematische-werke-von-karl-weierstrass-herau
     sgegeben-unter-][886] (retrieved 18-Feb-2019).
143. [WhiteheadRussell] Whitehead, Alfred North, and Bertrand Russell, *Principia Mathematica to
     *56,* Cambridge University Press, Cambridge, 1962 [QA9.W592 1962]; available at
     [https://ia600602.us.archive.org/35/items/PrincipiaMathematicaVolumeI/WhiteheadRussell-Principi
     aMathematicaVolumeI_text.pdf][887] (retrieved 22-Jan-2018).


Browsers and Fonts

Browsing with the Unicode font By default, the Metamath Proof Explorer uses a Unicode font for math
symbols. Some browsers may render Unicode incorrectly, although the problem is becoming less
frequent. On each page with a proof, at the top there is a link to switch to the GIF version. You
will stay in the GIF directory (you will see "mpegif" instead of "mpeuni" in the URL) until you
switch back to the Unicode font version on a page with a proof.

If you know of any browsers that work correctly with Metamath's Unicode symbols, let me know so that
I can put the information here. To check for correctness, compare the symbols in Unicode and GIF
versions of the [ASCII token chart][888].

Display font Since some people spend a great deal of time studying this site, I purposely left the
main font unspecified so that you can choose the one you find the most comfortable. In Firefox,
select Tools -> Options -> Content -> Default font. In Internet Explorer 11, select (gear symbol) ->
Internet Options -> General -> Fonts. On Windows, Times New Roman is the standard font. If you
prefer a sans-serif font, Arial will align correctly with our math symbol GIF images. By the way,
the minty green background color (#EEFFFA=r238,g255,b250) used for the proofs was chosen because it
has been claimed to be less fatiguing for long periods of work, but most browsers will let you
override it if you wish.

Viewing with a text browser The GIF version of the Metamath Proof Explorer can be viewed with a
text-only browser. All symbols will be shown as the [ASCII tokens][889] shown in the set.mm source
file. The most common text browser, Lynx, does not format tables, making proofs somewhat confusing
to read. But good results are achieved with the table-formatting text browser [w3m][890] [retrieved
21-Dec-2016].

One advantage of a text browser is that you can select and copy the formulas in a convenient ASCII
form that can be pasted into text documents, emails, etc. (The Mozilla browser will also do this if
the copy buffer is pasted into a text editor.) Text browsers are also extremely fast if you have a
slow connection. If you are blind, a text browser can make this site accessible, although for
theorem and proof displays, direct use of the Metamath program CLI (command-line interface) might be
more efficient—I would be interested in any feedback on this.

─┬─────────────────────────────────────────────────────────────────────────────┬────────────────────
 │*This page was last updated on 14-Jan-2024.*                                 │[W3C validator][893]
 │Your comments are welcome: Norman Megill [[nm at alum dot mit dot edu]][891] │[ Mobile test][894] 
 │Copyright terms: [Public domain][892]                                        │                    
─┴─────────────────────────────────────────────────────────────────────────────┴────────────────────

[1]: ../index.html
[2]: wn.html
[3]: grothprim.html
[4]: ../mm.html
[5]: ../index.html
[6]: mmtheorems.html
[7]: mmrecent.html
[8]: aleph0.html
[9]: ../index.html
[10]: ../index.html#contribute
[11]: #overview
[12]: #proofs
[13]: #axioms
[14]: #scaxioms
[15]: #pcaxioms
[16]: #staxioms
[17]: #groth
[18]: #class
[19]: #theorems
[20]: #trivia
[21]: #2p2e4length
[22]: #axiomnote
[23]: #traditional
[24]: #distinct
[25]: #dv-history
[26]: #dv-notes
[27]: #definitions
[28]: #assume
[29]: #function
[30]: #subsys
[31]: #oldaxioms
[32]: #read
[33]: #bib
[34]: #textonly
[35]: mmtheorems.html
[36]: mmrecent.html
[37]: https://us.metamath.org/mpeuni/mmrecent.html
[38]: conventions.html
[39]: mmbiblio.html
[40]: mmdefinitions.html
[41]: mmnatded.html
[42]: mmdeduction.html
[43]: mmcomplex.html
[44]: mmtopstr.html
[45]: mmfrege.html
[46]: mmzfcnd.html
[47]: mmascii.html
[48]: mmnotes.txt
[49]: ../index.html#contribute
[50]: https://github.com/metamath/set.mm
[51]: http://www.google.com/
[52]: ../index.html#mmprog
[53]: mmascii.html
[54]: pm54.43.html
[55]: https://www.youtube.com/watch?v=8WH4Rd4UKGE
[56]: #trivia
[57]: #trivia
[58]: #traditional
[59]: #distinct
[60]: https://www.youtube.com/watch?v=8WH4Rd4UKGE&t=7m32s
[61]: _mmbrows2p2e4.png
[62]: 2p2e4.html
[63]: ../index.html#pink
[64]: oveq2i.html
[65]: oveq2i.html
[66]: oveq2i.html
[67]: oveq2i.html
[68]: 2p2e4.html
[69]: df-ov.html
[70]: aleph1re.html
[71]: idALT.html
[72]: idALT.html
[73]: idALT.html
[74]: ax-1.html
[75]: idALT.html
[76]: c1.html
[77]: caddc.html
[78]: co.html
[79]: ../index.html#mmprog
[80]: df-2.html
[81]: #distinct
[82]: #class
[83]: ../index.html#start
[84]: ../downloads/metamath.pdf
[85]: #class
[86]: mmascii.html
[87]: #class
[88]: ax-12.html
[89]: ax-1.html
[90]: ax-mp.html
[91]: idALT.html
[92]: id.html
[93]: a1i.html
[94]: a1ii.html
[95]: id.html
[96]: id.html
[97]: pm2.27.html
[98]: ax-1.html
[99]: ax-2.html
[100]: ax-3.html
[101]: ax-mp.html
[102]: equid.html
[103]: #axiomnote
[104]: #traditional
[105]: #schemes
[106]: ax-gen.html
[107]: ax-4.html
[108]: ax-5.html
[109]: ax-6.html
[110]: ax-7.html
[111]: ax-8.html
[112]: ax-9.html
[113]: ax-10.html
[114]: ax-11.html
[115]: ax-12.html
[116]: ax-13.html
[117]: ax-5.html
[118]: #distinct
[119]: #wffsyntax
[120]: #compare
[121]: #oldaxioms
[122]: df-bi.html
[123]: df-an.html
[124]: df-ex.html
[125]: #definitions
[126]: df-bi.html
[127]: df-an.html
[128]: df-ex.html
[129]: wb.html
[130]: wa.html
[131]: wex.html
[132]: ru.html
[133]: axexte.html
[134]: axsep.html
[135]: axnul.html
[136]: axpr.html
[137]: #distinct
[138]: mmzfcnd.html
[139]: ax-ext.html
[140]: ax-rep.html
[141]: ax-pow.html
[142]: ax-un.html
[143]: ax-reg.html
[144]: ax-inf.html
[145]: ax-ac.html
[146]: ax-inf.html
[147]: ax-ac.html
[148]: ax-inf2.html
[149]: ax-ac2.html
[150]: ax-reg.html
[151]: ax-inf2.html
[152]: ax-inf.html
[153]: ax-ac.html
[154]: ax-ac2.html
[155]: ax-reg.html
[156]: grothprim.html
[157]: grothomex.html
[158]: grothac.html
[159]: grothpw.html
[160]: ax-groth.html
[161]: #figure1
[162]: ru.html
[163]: abid2.html
[164]: df-un.html
[165]: unjust.html
[166]: #figure1
[167]: df-2.html
[168]: df-1.html
[169]: df-add.html
[170]: df-clab.html
[171]: dfcleq.html
[172]: df-clel.html
[173]: df-clab.html
[174]: df-clab.html
[175]: df-sb.html
[176]: df-cleq.html
[177]: dfcleq.html
[178]: df-cleq.html
[179]: ax-ext.html
[180]: ax-ext.html
[181]: dfcleq.html
[182]: ax-ext.html
[183]: df-clel.html
[184]: df-cleq.html
[185]: cleljust.html
[186]: cvjust.html
[187]: cv.html
[188]: #distinct
[189]: dfcleq.html
[190]: df-clab.html
[191]: df-sb.html
[192]: eqabb.html
[193]: #Levy
[194]: #Quine
[195]: #TakeutiZaring
[196]: #Levy
[197]: #Quine
[198]: onprc.html
[199]: ncanth.html
[200]: ru.html
[201]: #Quine
[202]: df-an.html
[203]: #definitions
[204]: df-bi.html
[205]: mmtheorems.html
[206]: mmbiblio.html
[207]: ../mm_100.html
[208]: https://www.youtube.com/watch?v=LVGSeDjWzUo
[209]: idALT.html
[210]: peirce.html
[211]: anim12.html
[212]: exmid.html
[213]: pm5.18.html
[214]: consensus.html
[215]: meredith.html
[216]: 19.12.html
[217]: 19.35.html
[218]: equid.html
[219]: df-sb.html
[220]: 2eu5.html
[221]: uncom.html
[222]: eqabb.html
[223]: isset.html
[224]: df-op.html
[225]: ru.html
[226]: df-fv.html
[227]: canth.html
[228]: findes.html
[229]: peano1.html
[230]: peano2.html
[231]: peano3.html
[232]: peano4.html
[233]: peano5.html
[234]: omex.html
[235]: onprc.html
[236]: tfindes.html
[237]: tfr1.html
[238]: tfr2.html
[239]: tfr3.html
[240]: df-rdg.html
[241]: sbth.html
[242]: php.html
[243]: inf5.html
[244]: ac2.html
[245]: weth.html
[246]: zorn2.html
[247]: gch-kn.html
[248]: mmcomplex.html
[249]: arch.html
[250]: nn0opthi.html
[251]: sqrt2irr.html
[252]: nthruc.html
[253]: replimi.html
[254]: abstrii.html
[255]: eulerid.html
[256]: infpn.html
[257]: ruc.html
[258]: 2p2e4.html
[259]: 2cn.html
[260]: 2re.html
[261]: 1re.html
[262]: axrrecex.html
[263]: recexsr.html
[264]: sqgt0sr.html
[265]: pn0sr.html
[266]: distrsr.html
[267]: dmaddsr.html
[268]: addclsr.html
[269]: addsrpr.html
[270]: enrer.html
[271]: addcanpr.html
[272]: ltapr.html
[273]: ltaprlem.html
[274]: ltexpri.html
[275]: ltexprlem7.html
[276]: ltaddpr.html
[277]: addclpr.html
[278]: addclprlem2.html
[279]: addclprlem1.html
[280]: ltrnq.html
[281]: dmrecnq.html
[282]: recclnq.html
[283]: recidnq.html
[284]: recmulnq.html
[285]: mulassnq.html
[286]: mulerpq.html
[287]: nqereq.html
[288]: nqercl.html
[289]: nqerf.html
[290]: nqereu.html
[291]: enqer.html
[292]: mulcanpi.html
[293]: nnmcan.html
[294]: nnmword.html
[295]: nnmord.html
[296]: nnmordi.html
[297]: nnaword1.html
[298]: nnaword.html
[299]: nnaord.html
[300]: nnaordi.html
[301]: nnasuc.html
[302]: onasuc.html
[303]: frsuc.html
[304]: rdgsucg.html
[305]: rdgdmlim.html
[306]: tfr1a.html
[307]: tfrlem16.html
[308]: tfrlem15.html
[309]: tfrlem13.html
[310]: tfrlem12.html
[311]: tfrlem11.html
[312]: tfrlem10.html
[313]: tfrlem7.html
[314]: tfrlem5.html
[315]: tfrlem2.html
[316]: tfrlem1.html
[317]: fvreseq.html
[318]: eqfnfv.html
[319]: mpteqb.html
[320]: fvmpt2.html
[321]: fvmpt2i.html
[322]: fvmpti.html
[323]: fvmptg.html
[324]: fvopab3ig.html
[325]: funopfv.html
[326]: funbrfv.html
[327]: funeu.html
[328]: funmo.html
[329]: dffun6.html
[330]: dffun6f.html
[331]: dffun3.html
[332]: dffun2.html
[333]: brcnv.html
[334]: brcnvg.html
[335]: opelcnvg.html
[336]: brabg.html
[337]: brabga.html
[338]: opelopabga.html
[339]: copsex2g.html
[340]: copsexg.html
[341]: eqvinop.html
[342]: opth2.html
[343]: opthg2.html
[344]: opthg.html
[345]: opth.html
[346]: opeq1d.html
[347]: opeq1.html
[348]: ifbieq1d.html
[349]: ifeq1d.html
[350]: ifeq1.html
[351]: dfif6.html
[352]: uneq12i.html
[353]: uneq12.html
[354]: uneq2.html
[355]: uncom.html
[356]: uneqri.html
[357]: elun.html
[358]: elab2g.html
[359]: elabg.html
[360]: elabgf.html
[361]: vtoclgf.html
[362]: issetf.html
[363]: nfeq2.html
[364]: nfeq.html
[365]: nfcri.html
[366]: nfcrii.html
[367]: hblem.html
[368]: clelsb1.html
[369]: sbco2.html
[370]: sbied.html
[371]: sbie.html
[372]: sbbi.html
[373]: sban.html
[374]: sbim.html
[375]: sbi2.html
[376]: sbn.html
[377]: dfsb3.html
[378]: dfsb2.html
[379]: sb4.html
[380]: equs5.html
[381]: axc15.html
[382]: ax13a2.html
[383]: ax13v2.html
[384]: dveeq2.html
[385]: ax12lem3.html
[386]: 19.36v.html
[387]: 19.36.html
[388]: 19.9.html
[389]: 19.9h.html
[390]: 19.9t.html
[391]: 19.9ht.html
[392]: a10e.html
[393]: axc7.html
[394]: sp.html
[395]: 19.8a.html
[396]: equcoms.html
[397]: equcomi.html
[398]: equid.html
[399]: eximii.html
[400]: eximi.html
[401]: exim.html
[402]: alnex.html
[403]: con2bii.html
[404]: con1bii.html
[405]: xchbinx.html
[406]: notbii.html
[407]: notbi.html
[408]: notbid.html
[409]: 3bitr3g.html
[410]: syl5bbr.html
[411]: syl5bb.html
[412]: bitrd.html
[413]: bitri.html
[414]: sylibr.html
[415]: biimpri.html
[416]: bicomi.html
[417]: bicom1.html
[418]: bi2.html
[419]: dfbi1.html
[420]: impbii.html
[421]: bi3.html
[422]: simprim.html
[423]: impi.html
[424]: con1i.html
[425]: nsyl2.html
[426]: mt3d.html
[427]: con1d.html
[428]: notnot1.html
[429]: con2i.html
[430]: nsyl3.html
[431]: mt2d.html
[432]: con2d.html
[433]: notnot2.html
[434]: pm2.18d.html
[435]: pm2.18.html
[436]: pm2.21.html
[437]: pm2.21d.html
[438]: a1d.html
[439]: syl.html
[440]: mpd.html
[441]: a2i.html
[442]: ax-2.html
[443]: ../index.html#mmprog
[444]: mmcomplex.html#axioms
[445]: #2p2e4length
[446]: mmcomplex.html#axioms
[447]: dju1p1e2.html
[448]: #Tarski
[449]: #bibmegill
[450]: #traditional
[451]: #distinct
[452]: #Tarski
[453]: #scaxioms
[454]: #bibmegill
[455]: #wffsyntax
[456]: #traditional
[457]: #traditional
[458]: ax-c5.html
[459]: ax-1.html
[460]: #simplemetalogic
[461]: ax-gen.html
[462]: ax-4.html
[463]: ax-5.html
[464]: ax-6.html
[465]: ax-7.html
[466]: ax-8.html
[467]: ax-9.html
[468]: ax-10.html
[469]: ax-11.html
[470]: ax-12.html
[471]: ax-13.html
[472]: ax-10.html
[473]: ax-13.html
[474]: ax10w.html
[475]: ax11w.html
[476]: ax12w.html
[477]: ax13w.html
[478]: ax12wdemo.html
[479]: ax12w.html
[480]: ax-6.html
[481]: ax-6.html
[482]: ax6.html
[483]: ax-6.html
[484]: ax6v.html
[485]: ax-7.html
[486]: ax-8.html
[487]: ax-9.html
[488]: #axiomnotedv
[489]: ax-6.html
[490]: ax6e.html
[491]: ax6e.html
[492]: ax-mp.html
[493]: ax-gen.html
[494]: mmtheorems.html#ax6dgen
[495]: 2p2e4.html
[496]: #trivia
[497]: 1kp2ke3k.html
[498]: #distinct
[499]: dtru.html
[500]: dtru.html
[501]: dtru.html
[502]: dtru.html
[503]: #bibmegill
[504]: axc14.html
[505]: axc16.html
[506]: ax-c10.html
[507]: ax-6.html
[508]: ax6fromc10.html
[509]: sbequ.html
[510]: sbcom2.html
[511]: sbid2v.html
[512]: ax-6.html
[513]: https://web.archive.org/web/20130114072059/http://wiki.planetmath.org/cgi-bin/wiki.pl/Princip
al_instances_of_metatheorems
[514]: https://web.archive.org/web/20130203222517/http://wiki.planetmath.org/cgi-bin/wiki.pl/Distinc
tors_vs_binders
[515]: https://groups.google.com/d/msg/metamath/hJupn_hvqu8/KJg69J8BFAEJ
[516]: mmtheorems.html#ax6dgen
[517]: #KalishMontague
[518]: equid1.html
[519]: ../mmsolitaire/mms.html
[520]: ../award2003.html#16
[521]: ax-6.html
[522]: #distinct
[523]: #schemes
[524]: ax-1.html
[525]: ax-2.html
[526]: ax-3.html
[527]: ax-mp.html
[528]: stdpc4.html
[529]: stdpc5.html
[530]: ax-gen.html
[531]: stdpc6.html
[532]: stdpc7.html
[533]: #Tarski
[534]: http://www.appstate.edu/~hirstjl/primer/hirst.pdf
[535]: http://euclid.trentu.ca/math/sb/pcml/
[536]: #schemes
[537]: #pcaxioms
[538]: #Tarski
[539]: #pcaxioms
[540]: #simplemetalogic
[541]: hbequid.html
[542]: df-nf.html
[543]: df-sb.html
[544]: stdpc4.html
[545]: mmdeduction.html
[546]: natded.html
[547]: #axiomnote
[548]: ax-5.html
[549]: dtru.html
[550]: cleljustALT.html
[551]: ax-5.html
[552]: ax-5.html
[553]: cleljustALT.html
[554]: ax-13.html
[555]: ax-5.html
[556]: ax-5.html
[557]: ax-13.html
[558]: a1i.html
[559]: #schemes
[560]: #axiomnotedv
[561]: ax-5.html
[562]: ax-5.html
[563]: ax-5.html
[564]: ax13w.html
[565]: ax-13.html
[566]: ax13w.html
[567]: ax-5.html
[568]: cleljustALT.html
[569]: ../index.html#mmprog
[570]: ../index.html#mmj2
[571]: ../mmsolitaire/mms.html#q7
[572]: isset.html
[573]: #traditional
[574]: #Tarski
[575]: #Tarski
[576]: #Tarski
[577]: ax-5.html
[578]: ax6v.html
[579]: ax-6.html
[580]: ../other.html#hmm
[581]: #dvnote5
[582]: axsep.html
[583]: mpteq2ia.html
[584]: copsexg.html
[585]: dftr5.html
[586]: axsep.html
[587]: ralcom.html
[588]: ralcom2.html
[589]: vtoclef.html
[590]: dtru.html
[591]: #dtru
[592]: ../mmsolitaire/mms.html#q7
[593]: ../index.html#mmj2
[594]: ax6v.html
[595]: mmzfcnd.html#distinctors
[596]: dvdemo1.html
[597]: dvdemo2.html
[598]: df-2.html
[599]: ../index.html#mmj2
[600]: conventions.html
[601]: df-bi.html
[602]: df-clab.html
[603]: df-cleq.html
[604]: df-clel.html
[605]: #class
[606]: df-rdg.html
[607]: df-4.html
[608]: df-5.html
[609]: mmdefinitions.html
[610]: ../index.html#mmprog
[611]: tfr2.html
[612]: ax-ac.html
[613]: tfr2.html
[614]: ax-rep.html
[615]: tfr2.html
[616]: ax-rep.html
[617]: axrep1.html
[618]: axrep2.html
[619]: axrep3.html
[620]: axrep4.html
[621]: funimaexg.html
[622]: resfunexg.html
[623]: fnex.html
[624]: funex.html
[625]: tfrlem14.html
[626]: tfr1.html
[627]: ondomon.html
[628]: hartogs.html
[629]: ondomon.html
[630]: ax-ac2.html
[631]: hartogs.html
[632]: ax-ac2.html
[633]: cfv.html
[634]: ndmfv.html
[635]: sqrtthi.html
[636]: csqrt.html
[637]: cjval.html
[638]: facnn2.html
[639]: df-neg.html
[640]: negeq.html
[641]: fveq2.html
[642]: df-ov.html
[643]: cun.html
[644]: caddc.html
[645]: df-fn.html
[646]: ax-c4.html
[647]: ax-c5.html
[648]: ax-c7.html
[649]: ax-c9.html
[650]: ax-c10.html
[651]: ax-c11.html
[652]: ax-c11n.html
[653]: ax-c14.html
[654]: ax-c15.html
[655]: ax-c16.html
[656]: #pcaxioms
[657]: ax-c5.html
[658]: axc5.html
[659]: ax-4.html
[660]: ax-13.html
[661]: #bibmegill
[662]: ax-c4.html
[663]: ax-c11n.html
[664]: ax-c11.html
[665]: axc11n.html
[666]: #schemes
[667]: ax-c5.html
[668]: ax-c4.html
[669]: ax-c7.html
[670]: ax-c10.html
[671]: ax-c11.html
[672]: ax-c11n.html
[673]: ax-c15.html
[674]: ax-c9.html
[675]: ax-c14.html
[676]: ax-c16.html
[677]: axc5.html
[678]: axc4.html
[679]: axc7.html
[680]: axc10.html
[681]: axc11.html
[682]: axc11n.html
[683]: axc15.html
[684]: axc9.html
[685]: axc14.html
[686]: axc16.html
[687]: #axiomnote
[688]: #bibmegill
[689]: ax-5.html
[690]: ax5ALT.html
[691]: ax-c5.html
[692]: ax-c15.html
[693]: ax-c14.html
[694]: ax-5.html
[695]: ax-4.html
[696]: ax-5.html
[697]: ax-6.html
[698]: ax-7.html
[699]: ax-8.html
[700]: ax-9.html
[701]: ax-gen.html
[702]: ax-10.html
[703]: ax-11.html
[704]: ax-12.html
[705]: ax-13.html
[706]: ax-c11n.html
[707]: ax-12.html
[708]: ax-c15.html
[709]: ax-c16.html
[710]: ax-5.html
[711]: #bibmegill
[712]: mmzfcnd.html#distinctors
[713]: ax-c15.html
[714]: ax-12.html
[715]: ax-c15.html
[716]: ax-c16.html
[717]: ax-5.html
[718]: ax-c15.html
[719]: ../award2003.html#9b
[720]: ax-12.html
[721]: ax-c16.html
[722]: ax-5.html
[723]: ax-12.html
[724]: ax12fromc15.html
[725]: ax-c11.html
[726]: ax-12.html
[727]: ax-c11.html
[728]: ax-c5.html
[729]: ax-4.html
[730]: ax-10.html
[731]: ax-11.html
[732]: ax-gen.html
[733]: ax-4.html
[734]: ax-11.html
[735]: ax-mp.html
[736]: ax-gen.html
[737]: #traditional
[738]: ax-7.html
[739]: ax-5.html
[740]: ax-11.html
[741]: ../award2003.html#12
[742]: ax-4.html
[743]: ax-10.html
[744]: ax-c4.html
[745]: ax-c7.html
[746]: ax4fromc4.html
[747]: ax10fromc7.html
[748]: axc4.html
[749]: axc7.html
[750]: axc5c711.html
[751]: ax-c5.html
[752]: ax-c7.html
[753]: ax-11.html
[754]: #pcaxioms
[755]: https://groups.google.com/d/msg/metamath/OS4I_A32RTQ/6J2DmsJWCwAJ
[756]: https://github.com/metamath/set.mm/issues/703
[757]: #bibmegill
[758]: ax-c4.html
[759]: ax-c11n.html
[760]: ax-c11.html
[761]: axc11n.html
[762]: ax-c4.html
[763]: ax-c16.html
[764]: ax-4.html
[765]: ax-13.html
[766]: ax-c4.html
[767]: ax-c16.html
[768]: #bibmegill
[769]: ax-c4.html
[770]: ax-c16.html
[771]: ax-4.html
[772]: ax-13.html
[773]: #bibmegill
[774]: #bibmegill
[775]: ax-4.html
[776]: ax4fromc4.html
[777]: ax-5.html
[778]: ax-6.html
[779]: ax6fromc10.html
[780]: ax-7.html
[781]: ax-8.html
[782]: ax-9.html
[783]: ax-10.html
[784]: ax10fromc7.html
[785]: ax-11.html
[786]: ax-12.html
[787]: ax12fromc15.html
[788]: ax-13.html
[789]: ax13fromc9.html
[790]: ax-c4.html
[791]: axc4.html
[792]: ax-c5.html
[793]: axc5.html
[794]: ax-c7.html
[795]: axc7.html
[796]: ax-c9.html
[797]: axc9.html
[798]: ax-c10.html
[799]: axc10.html
[800]: ax-c11.html
[801]: axc11.html
[802]: ax-c11n.html
[803]: axc11n.html
[804]: ax-c14.html
[805]: axc14.html
[806]: ax-c15.html
[807]: axc15.html
[808]: ax-c16.html
[809]: axc16.html
[810]: mmbiblio.html
[811]: 19.35.html
[812]: #traditional
[813]: #class
[814]: mmbiblio.html
[815]: http://katmat.math.uni-bremen.de/acc
[816]: http://public.csusm.edu/aitken_html/m410/betweenness.08.pdf
[817]: http://public.csusm.edu/aitken_html/m410/
[818]: http://public.csusm.edu/aitken_html/m410/congruence.08.pdf
[819]: http://public.csusm.edu/aitken_html/m410/neutral.10.pdf
[820]: http://public.csusm.edu/aitken_html/m410/euclidean.10.pdf
[821]: http://public.csusm.edu/aitken_html/m410/quad.08.pdf
[822]: http://public.csusm.edu/aitken_html/m410/defectzero.08.pdf
[823]: http://public.csusm.edu/aitken_html/m410/conditions.08.pdf
[824]: http://public.csusm.edu/aitken_html/m410/hyperbolic.08.pdf
[825]: http://dx.doi.org/10.1090/bull/1556
[826]: https://www.researchgate.net/publication/304350412_Finding_Proofs_in_Tarskian_Geometry
[827]: http://www.michaelbeeson.com/research/papers/Tarski-JAR.pdf
[828]: https://philpapers.org/rec/BOBSL
[829]: http://www.springer.com/us/book/9783540225256
[830]: http://www.springer.com/us/book/9783540642435
[831]: https://link.springer.com/book/10.1007/978-3-642-61698-3
[832]: https://link.springer.com/book/9783540642398
[833]: http://www.springer.com/us/book/9783540642411
[834]: http://www.springer.com/us/book/9783540645634
[835]: http://www.springer.com/us/book/9783540653400
[836]: http://www.ams.org/journals/proc/1981-081-01/S0002-9939-1981-0589143-8/
[837]: http://www.danielclemente.com/logica/dn.en.pdf
[838]: http://www.danielclemente.com/logica/dn.en.html
[839]: http://www.math.miami.edu/~ec/book
[840]: http://diestel-graph-theory.com/index.html
[841]: https://archive.org/details/thirteenbookseu02heibgoog
[842]: http://aleph0.clarku.edu/~djoyce/java/elements/
[843]: https://archive.org/details/thirteenbookseu00heibgoog
[844]: https://archive.org/details/thirteenbookseu03heibgoog
[845]: https://gallica.bnf.fr/ark:/12148/bpt6k65658c
[846]: https://wiki.math.ntnu.no/_media/tma4225/2011/fremlin-mt1.pdf
[847]: https://www.essex.ac.uk/maths/people/fremlin/mt.htm
[848]: http://www.numdam.org/item/PMIHES_1960__4__5_0.pdf
[849]: http://www.math.cornell.edu/~hatcher/AT/ATpage.html
[850]: http://joshua.smcvt.edu/linearalgebra
[851]: https://arxiv.org/abs/1312.7748
[852]: https://webusers.imj-prg.fr/~harald.helfgott/anglais/book.html
[853]: http://www.humanities.mcmaster.ca/~hitchckd/peculiarities.pdf
[854]: http://arxiv.org/abs/math/9504224v1
[855]: https://randall-holmes.github.io/head.pdf
[856]: https://www.researchgate.net/publication/268313555_The_Friendship_Theorem
[857]: https://www.williamstein.org/edu/Spring2003/21n/papers/hilbert10.pdf
[858]: https://publications.lakeforest.edu/seniortheses/62/
[859]: http://gdz.sub.uni-goettingen.de/dms/load/img/?PID=GDZPPN002043211
[860]: http://math.bu.edu/people/aki/7.pdf
[861]: https://www.jstor.org/stable/2975081
[862]: https://arxiv.org/abs/1712.03500
[863]: http://www.scielo.org.mx/pdf/trf/n50/n50a1.pdf
[864]: http://xxx.lanl.gov/abs/quant-ph/0009039
[865]: http://projecteuclid.org/euclid.ndjfl/1040149359
[866]: ../downloads/finiteaxiom.pdf
[867]: #technote
[868]: http://xxx.lanl.gov/abs/quant-ph/0009038
[869]: http://xxx.lanl.gov/abs/quant-ph/0108074
[870]: http://community.dur.ac.uk/george.mertzios/papers/Conf/Conf_Windmills.pdf
[871]: http://euclid.colorado.edu/~monkd/monk07.pdf
[872]: http://gdz.sub.uni-goettingen.de/dms/load/img/?PID=GDZPPN00204322X
[873]: https://www.ams.org/journals/mcom/2014-83-288/S0025-5718-2013-02787-1/home.html
[874]: http://www.cs.cmu.edu/~fp/courses/atp/handouts/atp.pdf
[875]: http://web.archive.org/web/20160304013704/http://www.cs.cmu.edu/~fp/courses/atp/handouts/atp.
pdf
[876]: https://projecteuclid.org/journals/illinois-journal-of-mathematics/volume-6/issue-1/Approxima
te-formulas-for-some-functions-of-prime-numbers/10.1215/ijm/1255631807.full
[877]: https://books.google.com/books?id=h_AUynB6PA8C&pg=PA39#v=onepage&q&f=false
[878]: https://jjsch.github.io/output/oa.pdf
[879]: https://eclass.uoa.gr/modules/document/file.php/MATH594/Stewart%20Galois%204th%20edition.pdf
[880]: http://ocw.mit.edu/resources/res-18-001-calculus-online-textbook-spring-2005/
[881]: http://www.randomservices.org/random/index.html
[882]: http://gdz.sub.uni-goettingen.de/dms/load/img/?PID=GDZPPN002043203
[883]: http://www.math.uiuc.edu/~vddries/recursion.pdf
[884]: https://ocw.mit.edu/courses/mathematics/18-125-measure-and-integration-fall-2003/lecture-note
s/18125_lec7.pdf
[885]: https://ocw.mit.edu/courses/mathematics/18-125-measure-and-integration-fall-2003/lecture-note
s/18125_lec8.pdf
[886]: https://www.yumpu.com/de/document/read/20688945/mathematische-werke-von-karl-weierstrass-hera
usgegeben-unter-
[887]: https://ia600602.us.archive.org/35/items/PrincipiaMathematicaVolumeI/WhiteheadRussell-Princip
iaMathematicaVolumeI_text.pdf
[888]: mmascii.html
[889]: mmascii.html
[890]: http://w3m.sourceforge.net/
[891]: ../email.html
[892]: ../copyright.html#pd
[893]: https://validator.w3.org/check?uri=referer
[894]: https://www.google.com/webmasters/tools/mobile-friendly/
