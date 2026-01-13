# Crate stateright Copy item path

[Source][1]
Expand description

A library for model checking systems, with an emphasis on distributed systems.

Most of the docs are here, but a short book is being written to provide a gentler introduction:
“[Building Distributed Systems With Stateright][2].”

## [§][3]Introduction to Model Checking

[`Model`][4] implementations indicate how a system evolves, such as a set of actors executing a
distributed protocol on an IP network. Incidentally, that scenario is so common for model checking
that Stateright includes an [`actor::ActorModel`][5], and unlike many model checkers, Stateright is
also able to [spawn][6] these actors on a real network.

Models of a system are supplemented with [`always`][7] and [`sometimes`][8] properties. An `always`
[`Property`][9] (also known as a [safety property][10] or [invariant][11]) indicates that a specific
problematic outcome is not possible, such as data inconsistency. A `sometimes` property on the other
hand is used to ensure a particular outcome is reachable, such as the ability for a distributed
system to process a write request.

A [`Checker`][12] will attempt to [discover][13] a counterexample for every `always` property and an
example for every `sometimes` property, and these examples/counterexamples are indicated by
sequences of system steps known as [`Path`][14]s (also known as traces or behaviors). The presence
of an `always` discovery or the absence of a `sometimes` discovery indicate that the model checker
identified a problem with the code under test.

## [§][15]Example

A toy example that solves a [sliding puzzle][16] follows. This simple example leverages only a
`sometimes` property, but in most cases for a real system you would have an `always` property at a
minimum. The imagined use case is one in which we must ensure that a particular configuration of the
puzzle has a solution.

**TIP**: see the [`actor`][17] module documentation for an actor system example. More sophisticated
examples are available in the [`examples/` directory of the repository][18]

`use stateright::*;

#[derive(Clone, Debug, Eq, PartialEq)]
enum Slide { Down, Up, Right, Left }

struct Puzzle([u8; 9]);
impl Model for Puzzle {
    type State = [u8; 9];
    type Action = Slide;

    fn init_states(&self) -> Vec<Self::State> {
        vec![self.0]
    }

    fn actions(&self, _state: &Self::State, actions: &mut Vec<Self::Action>) {
        actions.append(&mut vec![
            Slide::Down, Slide::Up, Slide::Right, Slide::Left
        ]);
    }

    fn next_state(&self, last_state: &Self::State, action: Self::Action) -> Option<Self::State> {
        let empty = last_state.iter().position(|x| *x == 0).unwrap();
        let empty_y = empty / 3;
        let empty_x = empty % 3;
        let maybe_from = match action {
            Slide::Down  if empty_y > 0 => Some(empty - 3), // above
            Slide::Up    if empty_y < 2 => Some(empty + 3), // below
            Slide::Right if empty_x > 0 => Some(empty - 1), // left
            Slide::Left  if empty_x < 2 => Some(empty + 1), // right
            _ => None
        };
        maybe_from.map(|from| {
            let mut next_state = *last_state;
            next_state[empty] = last_state[from];
            next_state[from] = 0;
            next_state
        })
    }

    fn properties(&self) -> Vec<Property<Self>> {
        vec![Property::<Self>::sometimes("solved", |_, state| {
            let solved = [0, 1, 2,
                          3, 4, 5,
                          6, 7, 8];
            state == &solved
        })]
    }
}
let checker = Puzzle([1, 4, 2,
                      3, 5, 8,
                      6, 7, 0])
    .checker().spawn_bfs().join();
checker.assert_properties();
checker.assert_discovery("solved", vec![
        Slide::Down,
        // ... results in:
        //       [1, 4, 2,
        //        3, 5, 0,
        //        6, 7, 8]
        Slide::Right,
        // ... results in:
        //       [1, 4, 2,
        //        3, 0, 5,
        //        6, 7, 8]
        Slide::Down,
        // ... results in:
        //       [1, 0, 2,
        //        3, 4, 5,
        //        6, 7, 8]
        Slide::Right,
        // ... results in:
        //       [0, 1, 2,
        //        3, 4, 5,
        //        6, 7, 8]
    ]);`

## [§][19]What to Read Next

The [`actor`][20] and [`semantics`][21] submodules will be of particular interest to most
individuals.

Also, as mentioned earlier, you can find [more examples][22] in the Stateright repository.

## Modules[§][23]

*[actor][24]*
  This module provides an [Actor][25] trait, which can be model checked using [`ActorModel`][26].
  You can also [`spawn()`][27] the actor in which case it will communicate over a UDP socket.
*[report][28]*
*[semantics][29]*
  This module provides code to define and verify the correctness of an object or system based on how
  it responds to a collection of potentially concurrent (i.e. partially ordered) operations.
*[util][30]*
  Utilities such as [`HashableHashSet`][31] and [`HashableHashMap`][32]. Those two in particular are
  useful because the corresponding [`HashSet`][33] and [`HashMap`][34] do not implement
  [`Hash`][35], meaning they cannot be used directly in models.

## Structs[§][36]

*[CheckerBuilder][37]*
  A [`Model`][38] [`Checker`][39] builder. Instantiable via the [`Model::checker`][40] method.
*[Path][41]*
  A path of states including actions. i.e. `state --action--> state ... --action--> state`.
*[PathRecorder][42]*
  A [`CheckerVisitor`][43] that records paths visited by the model checker.
*[Property][44]*
  A named predicate, such as “an epoch *sometimes* has no leader” (for which the model checker would
  find an example) or “an epoch *always* has at most one leader” (for which the model checker would
  find a counterexample) or “a proposal is *eventually* accepted” (for which the model checker would
  find a counterexample path leading from the initial state through to a terminal state).
*[RewritePlan][45]*
  A `RewritePlan<R>` is derived from a data structure instance and indicates how values of type `R`
  (short for “rewritten”) should be rewritten. When that plan is recursively applied via
  [`Rewrite`][46], the resulting data structure instance will be behaviorally equivalent to the
  original data structure under a symmetry equivalence relation, enabling symmetry reduction.
*[StateRecorder][47]*
  A [`CheckerVisitor`][48] that records states evaluated by the model checker. Does not record
  generated states that are still pending property evaluation.
*[UniformChooser][49]*
  A chooser that makes uniform choices.

## Enums[§][50]

*[DiscoveryClassification][51]*
  The classification of a property discovery.
*[Expectation][52]*
  Indicates whether a property is always, eventually, or sometimes true.
*[HasDiscoveries][53]*

## Traits[§][54]

*[Checker][55]*
  Implementations perform [`Model`][56] checking.
*[CheckerVisitor][57]*
  A visitor to apply to every [`Path`][58] of the checked [`Model`][59].
*[Chooser][60]*
  Choose transitions in the model.
*[Model][61]*
  This is the primary abstraction for Stateright. Implementations model a nondeterministic system’s
  evolution. If you are using Stateright’s actor framework, then you do not need to implement this
  interface and can instead leverage [`actor::Actor`][62] and [`actor::ActorModel`][63].
*[Representative][64]*
  This trait is used to reduce the state space when checking a model with
  [`CheckerBuilder::symmetry`][65]. The trait indicates the ability to generate a representative
  from a symmetry equivalence class for each state in [`Model::State`][66].
*[Rewrite][67]*
  Implementations can rewrite their instances of the “rewritten” type `R` based on a specified
  [`RewritePlan`][68].

[1]: ../src/stateright/lib.rs.html#1-390
[2]: https://www.stateright.rs
[3]: #introduction-to-model-checking
[4]: trait.Model.html
[5]: actor/struct.ActorModel.html
[6]: actor/fn.spawn.html
[7]: struct.Property.html#method.always
[8]: struct.Property.html#method.sometimes
[9]: struct.Property.html
[10]: https://en.wikipedia.org/wiki/Safety_property
[11]: https://en.wikipedia.org/wiki/Invariant_(computer_science)
[12]: trait.Checker.html
[13]: trait.Checker.html#tymethod.discoveries
[14]: struct.Path.html
[15]: #example
[16]: https://en.wikipedia.org/wiki/Sliding_puzzle
[17]: actor/index.html
[18]: https://github.com/stateright/stateright/tree/master/examples
[19]: #what-to-read-next
[20]: actor/index.html
[21]: semantics/index.html
[22]: https://github.com/stateright/stateright/tree/master/examples
[23]: #modules
[24]: actor/index.html
[25]: actor/trait.Actor.html
[26]: actor/struct.ActorModel.html
[27]: actor/fn.spawn.html
[28]: report/index.html
[29]: semantics/index.html
[30]: util/index.html
[31]: util/struct.HashableHashSet.html
[32]: util/struct.HashableHashMap.html
[33]: https://doc.rust-lang.org/nightly/std/collections/hash/set/struct.HashSet.html
[34]: https://doc.rust-lang.org/nightly/std/collections/hash/map/struct.HashMap.html
[35]: https://doc.rust-lang.org/nightly/core/hash/trait.Hash.html
[36]: #structs
[37]: struct.CheckerBuilder.html
[38]: trait.Model.html
[39]: trait.Checker.html
[40]: trait.Model.html#method.checker
[41]: struct.Path.html
[42]: struct.PathRecorder.html
[43]: trait.CheckerVisitor.html
[44]: struct.Property.html
[45]: struct.RewritePlan.html
[46]: trait.Rewrite.html
[47]: struct.StateRecorder.html
[48]: trait.CheckerVisitor.html
[49]: struct.UniformChooser.html
[50]: #enums
[51]: enum.DiscoveryClassification.html
[52]: enum.Expectation.html
[53]: enum.HasDiscoveries.html
[54]: #traits
[55]: trait.Checker.html
[56]: trait.Model.html
[57]: trait.CheckerVisitor.html
[58]: struct.Path.html
[59]: trait.Model.html
[60]: trait.Chooser.html
[61]: trait.Model.html
[62]: actor/trait.Actor.html
[63]: actor/struct.ActorModel.html
[64]: trait.Representative.html
[65]: struct.CheckerBuilder.html#method.symmetry
[66]: trait.Model.html#associatedtype.State
[67]: trait.Rewrite.html
[68]: struct.RewritePlan.html
