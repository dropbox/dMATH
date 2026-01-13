* [mCRL2 tutorial][1]
* A Vending Machine
* [ View page source][2]

# A Vending Machine[][3]

**Contribution of this section**

1. specifying processes,
2. linearisation,
3. state space exploration,
4. visualisation of state spaces,
5. comparison/reduction using behavioural equivalences, and
6. verification of modal mu-calculus properties.

**New tools**: [mcrl22lps][4], [lps2lts][5], [ltsgraph][6], [ltscompare][7], [ltsconvert][8],
[lps2pbes][9], [pbessolve][10].

Our first little step consists of number of variations on the good old vending machine, a user
`User` interacting with a machine `Mach`. By way of this example we will encounter the basic
ingredients of mCRL2. In the first variation of the vending machine, a very primitive machine and
user, are specified. Some properties are verified. In the second variation non-determinism is
considered and, additionally, some visualization and comparison tools from the toolset are
illustrated. The third variation comes closer to a rudimentary prototype specification.

## First variation[][11]

After inserting a coin of 10 cents, the user can push the button for an apple. An apple will then be
put in the drawer of the machine. See Figure [Vending machine 1][12].

[../../../_images/mach1.png]

Vending machine 1[][13]

Vending machine 1 can be specified by the following mCRL2, also included in the file
[`vm01.mcrl2`][14].

act
  ins10, optA, acc10, putA, coin, ready ;
proc
  User = ins10 . optA . User ;
  Mach = acc10 . putA . Mach ;
init
  allow(
    { coin, ready },
    comm(
      { ins10|acc10 -> coin, optA|putA -> ready },
      User || Mach
  ) ) ;

The specification is split in three sections:

1. `act`, a declaration of actions of 6 actions,
2. `proc`, the definition of 2 processes, and
3. `init`, the initialization of the system.

The process `User` is recursively defined as doing an `ins10` action, followed by an `optA` action,
followed by the process `User` again. The process `Mach` is similar, looping on the action `acc10`
followed by the action `putA`. Note, only four actions are used in the definition of the processes.
In particular, the action `coin` and `ready` are not referred to.

The initialization of the system has a typical form. A number of parallel processes, in the context
of a communication function, with a limited set of actions allowed. So, `||` is the parallel
operator, in this case putting the processes `User` and `Mach` in parallel. The communication
function is the first argument of the `comm` operator. Here, we have that synchronization of an
`ins10` action and an `acc10` action yields the action `coin`, whereas synchronization of `optA` and
`putA` yields `ready`. The actions of the system that are allowed, are mentioned in the first
argument of the allow operator `allow`. Thus, for our first system only `coin` and `ready` are
allowed actions.

We compile the specification in the file [`vm01.mcrl2`][15] to a so-called linear process, saved in
the file `vm01.lps`. This can be achieved by running:

$ mcrl22lps vm01.mcrl2 vm01.lps

on the command line. The linear process is the internal representation format of mCRL2, and is not
meant for human inspection. However, from `vm01.lps` a labeled transition system, LTS for short, can
be obtained by running:

$ lps2lts vm01.lps vm01.lts

which can be viewed by the [ltsgraph][16] facility, by typing:

$ ltsgraph vm01.lts

at the prompt. Some manual beautifying yields the picture in Figure [LTS of vending machine 1][17].

[../../../_images/vm01.png]

LTS of vending machine 1[][18]

Apparently, starting from state 0 the system shuttles between state 0 and 1 alternating the actions
`coin` and `ready`. Enforced by the allow operator, unmatched `ins10`, `acc10`, `optA` and `putA`
actions are excluded. The actions synchronize pairwise, `ins10` with `acc10`, `optA` with `putA`, to
produce `coin` and `ready`, respectively.

As a first illustration of model checking in mCRL2, we consider some simple properties to be checked
against the specification [`vm01.mcrl2`][19]. Given the LTS of the system, the properties obviously
hold.

1. [`vm01a.mcf`][20]:
   
   % always, eventually a ready is possible (true)
   
   [ true* ] < true* . ready > true
   
   In this property, `[true*]` represents all finite sequences of actions starting from the initial
   state. `<true*.ready>` expresses the existence of a sequence of actions ending with the action
   `ready`. The last occurence of `true` in this property is a logical formula to be evaluated in
   the current state. Thus, if the property is satisfied by the system, then after any finite
   sequence of actions, `[true*]`, the system can continue with some finite sequence of actions
   ending with `ready`, `<true*.ready>`, and reaches a state in which the formula `true` holds.
   Since `true` always holds, this property states that a next `ready` is always possible.
2. [`vm01b.mcf`][21]:
   
   % a ready is always possible (false)
   
   [ true* ] < ready > true
   
   This property is less liberal than property (a). Here, `<ready> true` requires a `ready` action
   to be possible for the system, after any finite sequence, `[true*]`. This property does not hold.
   A `ready` action is not immediately followed by a `ready` action again. Also, `ready` is not
   possible in the initial state.
3. [`vm01c.mcf`][22]:
   
   % after every ready only a coin follows (true)
   
   [ true* . ready . !coin ] false
   
   This property uses the complement construct. `!coin` are all actions different from `coin`. So,
   any sequence of actions with `ready` as its one but final action and ending with an action
   different from `coin`, leads to a state where `false` holds. Since no such state exists, there
   are no path of the form `true*.ready.!coin`. Thus, after any `ready` action, any action that
   follows, if any, will be `coin`.
4. [`vm01d.mcf`][23]:
   
   % any ready is followed by a coin and another ready (true)
   
   [ true* . ready . !coin ] false  &&  [ true* . ready . true . !ready ] false
   
   This property is a further variation involving conjunction `&&`.

Model checking with mCRL2 is done by constructing a so-called parameterised boolean equation system
or PBES from a linear process specification and a modal [\mu]-calculus formula. For example, to
verify property (a) above, we call the [lps2pbes][24] tool. Assuming property (a) to be in file
[`vm01a.mcf`][25], running:

$ lps2pbes vm01.lps -f vm01a.mcf vm01a.pbes

creates from the system in linear format and the formula in the file [`vm01.mcrl2`][26] right after
the `-f` switch, a PBES in the file `vm01a.pbes`. On calling the PBES solver on `vm01a.pbes`:

$ pbessolve vm01a.pbes

the mCRL2 tool answers:

true

So, for vending machine 1 it holds that action `ready` is always possible in the future. Instead of
making separate steps explicity, the verification can also be captured by a single, pipe-line
command:

$ mcrl22lps vm01.mcrl2 | lps2pbes -f vm01a.mcf | pbessolve

Running the other properties yields the expected results. Properties (c) and (d) do hold, property
(b) does not hold, as indicated by the following snippet:

$ mcrl22lps vm01.mcrl2 | lps2pbes -f vm01b.mcf | pbessolve
false
$ mcrl22lps vm01.mcrl2 | lps2pbes -f vm01c.mcf | pbessolve
true
$ mcrl22lps vm01.mcrl2 | lps2pbes -f vm01d.mcf | pbessolve
true

## Second variation[][27]

Next, we add a chocolate bar to the assortment of the vending machine. A chocolate bar costs 20
cents, an apple 10 cents. The machine will now accept coins of 10 and 20 cents. The scenarios
allowed are (i) insertion of 10 cent and purchasing an apple, (ii) insertion of 10 cent twice or 20
cent once and purchasing a chocolate bar. Additionally, after insertion of money, the user can push
the change button, after which the inserted money is returned. See Figure [Vending machine 2][28].

[../../../_images/mach2.png]

Vending machine 2[][29]

Exercise

Extend the following mCRL2 specification ([`vm02-holes.mcrl2`][30]) to describe the vending machine
sketched above, and save the resulting specification as `vm02.mcrl2`. The actions that are involved,
and a possible specification of the `Mach` process have been given. The machine is required to
perform a `prod` action for administration purposes.

act
  ins10, ins20, acc10, acc20, coin10, coin20, ret10, ret20 ;
  optA, optC, chg10, chg20, putA, putC, prod, 
  readyA, readyC, out10, out20 ;

proc
  User = 
    *1*

  Mach = 
    acc10.( putA.prod + acc10.( putC.prod + ret20 ) + ret10 ).Mach +
    acc20.( putA.prod.ret10 + putC.prod + ret20 ).Mach ;

init
  *2* ;

Linearise your specification using [mcrl22lps][31], saving the LPS as `vm02.lps`.

Solution

A sample solution is available in [`vm02.mcrl2`][32]. This can be linearised using:

$ mcrl22lps vm02.mcrl2 vm02.lps

A visualization of the specified system can be obtained by first converting the linear process into
a labeled transition system (in so-called SVC-format) by:

$  lps2lts vm02.lps vm02.lts

and next loading the SVC file `vm02.lts` into the ltsgraph tool by:

$  ltsgraph vm02.lts

The LTS can be beautified (a bit) using the `start` button in the optimization panel of the user
interface. Manual manipulation by dragging states is also possible. For small examples, increasing
the natural transition length may provide better results.

Exercise

Prove that your specification satisfies the following properties:

1. no three 10ct coins can be inserted in a row,
2. no chocolate after 10ct only, and
3. an apple only after 10ct, a chocolate after 20ct.

Solution

Each of the properties can be expressed as a µ-calculus formula. Possible solutions are given as
[`vm02a.mcf`][33], [`vm02b.mcf`][34], and [`vm02c.mcf`][35].

Each of the properties can be checked using a combination of [mcrl22lps][36], [lps2pbes][37] and
[pbessolve][38]. The following is a sample script that performs the verification:

$ mcrl22lps vm02.mcrl2 vm02.lps
$ lps2pbes vm02.lps -f vm02a.mcf | pbessolve
true
$ lps2pbes vm02.lps -f vm02b.mcf | pbessolve
true
$ lps2pbes vm02.lps -f vm02c.mcf | pbessolve
true

So the conclusion of the verification is that all three properties hold.

The file [`vm02-taus.mcrl2`][39] contains the specification of a system performing `coin10` and
`coin20` actions as well as so-called [\tau]-steps. Using the [ltscompare][40] tool you can compare
your model under branching bisimilarity with the LTS of the system `vm02-taus`, after hiding the
actions `readyA`, `readyC`, `out10`, `out20`, `prod` using the following command:

$ ltscompare -ebranching-bisim --tau=out10,out20,readyA,readyC,prod vm02.lts vm02-taus.lts

Note

You first need to generate the state space of [`vm02-taus.mcrl2`][41] using [mcrl22lps][42] and
[lps2lts][43].

Using [ltsconvert][44], the LTS for `vm02.mcrl2` can be minimized with respect to branching
bisimulation after hiding the readies and returns:

$ ltsconvert -ebranching-bisim --tau=out10,out20,readyA,readyC,prod vm02.lts vm02min.lts

Exercise

Compare the LTSs `vm02min.lts` and vm02-taus.lts visually using [ltsgraph][45].

## Third variation[][46]

A basic version of a vending machine with parametrized actions is available in the file
[`vm03-basic.mcrl2`][47].

Exercise

Modify this specification such that all coins of denomination 50ct, 20ct, 10ct and 5ct can be
inserted. The machine accumulates upto a total of 60 cents. If sufficient credit, an apple or
chocolate bar is supplied after selection. Money is returned after pressing the change button.

[ Previous][48] [Next ][49]

© Copyright 2011-2025, Technische Universiteit Eindhoven.

Built with [Sphinx][50] using a [theme][51] provided by [Read the Docs][52].

[1]: ../tutorial.html
[2]: ../../../_sources/user_manual/tutorial/machine/index.rst.txt
[3]: #a-vending-machine
[4]: ../../tools/release/mcrl22lps.html#tool-mcrl22lps
[5]: ../../tools/release/lps2lts.html#tool-lps2lts
[6]: ../../tools/release/ltsgraph.html#tool-ltsgraph
[7]: ../../tools/release/ltscompare.html#tool-ltscompare
[8]: ../../tools/release/ltsconvert.html#tool-ltsconvert
[9]: ../../tools/release/lps2pbes.html#tool-lps2pbes
[10]: ../../tools/release/pbessolve.html#tool-pbessolve
[11]: #first-variation
[12]: #fig-mach1
[13]: #id1
[14]: ../../../_downloads/007a9f75a88631ac8e05115e84b98fe1/vm01.mcrl2
[15]: ../../../_downloads/007a9f75a88631ac8e05115e84b98fe1/vm01.mcrl2
[16]: ../../tools/release/ltsgraph.html#tool-ltsgraph
[17]: #fig-lts-vm01
[18]: #id2
[19]: ../../../_downloads/007a9f75a88631ac8e05115e84b98fe1/vm01.mcrl2
[20]: ../../../_downloads/e621622b39d0b28514d3d4b5c5e004c2/vm01a.mcf
[21]: ../../../_downloads/4d18d42f795ff59c91c92bc33302a7dd/vm01b.mcf
[22]: ../../../_downloads/f0b765cd5f818cd25c608248a42a11df/vm01c.mcf
[23]: ../../../_downloads/f3d8300d902dc4f79405aa9e252f164a/vm01d.mcf
[24]: ../../tools/release/lps2pbes.html#tool-lps2pbes
[25]: ../../../_downloads/e621622b39d0b28514d3d4b5c5e004c2/vm01a.mcf
[26]: ../../../_downloads/007a9f75a88631ac8e05115e84b98fe1/vm01.mcrl2
[27]: #second-variation
[28]: #fig-mach2
[29]: #id3
[30]: ../../../_downloads/0070bbb33b71416bc6eda0d87fbc88c0/vm02-holes.mcrl2
[31]: ../../tools/release/mcrl22lps.html#tool-mcrl22lps
[32]: ../../../_downloads/e1ef5752227ac0c38d3311f5642bd6e9/vm02.mcrl2
[33]: ../../../_downloads/8a47e4fe21381f987c17df1c58ed132b/vm02a.mcf
[34]: ../../../_downloads/b326434818d14a686cd08f6c5fd443be/vm02b.mcf
[35]: ../../../_downloads/09902fe83a4a83d7306348d2bbe1c940/vm02c.mcf
[36]: ../../tools/release/mcrl22lps.html#tool-mcrl22lps
[37]: ../../tools/release/lps2pbes.html#tool-lps2pbes
[38]: ../../tools/release/pbessolve.html#tool-pbessolve
[39]: ../../../_downloads/2efa04125af3ea4efdd192cf8400603a/vm02-taus.mcrl2
[40]: ../../tools/release/ltscompare.html#tool-ltscompare
[41]: ../../../_downloads/2efa04125af3ea4efdd192cf8400603a/vm02-taus.mcrl2
[42]: ../../tools/release/mcrl22lps.html#tool-mcrl22lps
[43]: ../../tools/release/lps2lts.html#tool-lps2lts
[44]: ../../tools/release/ltsconvert.html#tool-ltsconvert
[45]: ../../tools/release/ltsgraph.html#tool-ltsgraph
[46]: #third-variation
[47]: ../../../_downloads/00e84ed393779280c3aaf763e4ae8273/vm03-basic.mcrl2
[48]: ../tutorial.html
[49]: ../watercans/index.html
[50]: https://www.sphinx-doc.org/
[51]: https://github.com/readthedocs/sphinx_rtd_theme
[52]: https://readthedocs.org
