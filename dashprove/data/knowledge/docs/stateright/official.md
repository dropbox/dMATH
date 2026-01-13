[[chat]][1] [[crates.io]][2] [[docs.rs]][3] [[LICENSE]][4]

Correctly implementing distributed algorithms such as the [Paxos][5] and [Raft][6] consensus
protocols is notoriously difficult due to inherent nondetermism such as message reordering by
network devices. Stateright is a [Rust][7] actor library that aims to solve this problem by
providing an embedded [model checker][8], a UI for exploring system behavior ([demo][9]), and a
lightweight actor runtime. It also features a linearizability tester that can be run within the
model checker for more exhaustive test coverage than similar solutions such as [Jepsen][10].

[[Stateright Explorer screenshot]][11]

## Getting Started

1. **Please see the book, "[Building Distributed Systems With Stateright][12]."**
2. A [video introduction][13] is also available.
3. Stateright also has detailed [API docs][14].
4. Consider also joining the [Stateright Discord server][15] for Q&A or other feedback.

## Examples

Stateright includes a variety of [examples][16], such as a [Single Decree Paxos cluster][17] and an
[abstract two phase commit model][18].

Passing a `check` CLI argument causes each example to validate itself using Stateright's model
checker:

# Two phase commit with 3 resource managers.
cargo run --release --example 2pc check 3
# Paxos cluster with 3 clients.
cargo run --release --example paxos check 3
# Single-copy (unreplicated) register with 3 clients.
cargo run --release --example single-copy-register check 3
# Linearizable distributed register (ABD algorithm) with 2 clients
# assuming ordered channels between actors.
cargo run --release --example linearizable-register check 2 ordered

Passing an `explore` CLI argument causes each example to spin up the Stateright Explorer web UI
locally on port 3000, allowing you to browse system behaviors:

cargo run --release --example 2pc explore
cargo run --release --example paxos explore
cargo run --release --example single-copy-register explore
cargo run --release --example linearizable-register explore

Passing a `spawn` CLI argument to the examples leveraging the actor functionality will cause each to
spawn actors using the included runtime, transmitting JSON messages over UDP:

cargo run --release --example paxos spawn
cargo run --release --example single-copy-register spawn
cargo run --release --example linearizable-register spawn

The `bench.sh` script runs all the examples with various settings for benchmarking the performance
impact of changes to the library.

./bench.sh

# Features

Stateright contains a general purpose model checker offering:

* Invariant checks via "always" properties.
* Nontriviality checks via "sometimes" properties.
* Liveness checks via "eventually" properties (experimental/incomplete).
* A web browser UI for interactively exploring state space.
* [Linearizability][19] and [sequential consistency][20] testers.
* Support for symmetry reduction to reduce state spaces.

Stateright's actor system features include:

* An actor runtime that can execute actors outside the model checker in the "real world."
* A model for lossy/lossless duplicating/non-duplicating networks with the ability to capture actor
  message [history][21] to check an actor system against an expected consistency model.
* Pluggable network semantics for model checking, allowing you to choose between fewer assumptions
  (e.g. "lossy unordered duplicating") or more assumptions (speeding up model checking; e.g.
  "lossless ordered").
* An optional network adapter that provides a lossless non-duplicating ordered virtual channel for
  messages between a pair of actors.

In contrast with other actor libraries, Stateright enables you to [formally verify][22] the
correctness of your implementation, and in contrast with model checkers such as TLC for [TLA+][23],
systems implemented using Stateright can also be run on a real network without being reimplemented
in a different language.

## Contribution

Contributions are welcome! Please [fork the library][24], push changes to your fork, and send a
[pull request][25]. All contributions are shared under an MIT license unless explicitly stated
otherwise in the pull request.

## License

Stateright is copyright 2018 Jonathan Nadal and other [contributors][26]. It is made available under
the MIT License.

To avoid the need for a Javascript package manager, the Stateright repository includes code for the
following Javascript dependency used by Stateright Explorer:

* [KnockoutJS][27] is copyright 2010 Steven Sanderson, the Knockout.js team, and other contributors.
  It is made available under the MIT License.

[1]: https://discord.gg/JbxGSVP4A6
[2]: https://crates.io/crates/stateright
[3]: https://docs.rs/stateright
[4]: https://github.com/stateright/stateright/blob/master/LICENSE
[5]: https://en.wikipedia.org/wiki/Paxos_%28computer_science%29
[6]: https://en.wikipedia.org/wiki/Raft_%28computer_science%29
[7]: https://www.rust-lang.org/
[8]: https://en.wikipedia.org/wiki/Model_checking
[9]: http://demo.stateright.rs:3000/
[10]: https://jepsen.io/
[11]: https://raw.githubusercontent.com/stateright/stateright/master/explorer.png
[12]: https://www.stateright.rs
[13]: https://youtube.com/playlist?list=PLUhyBsVvEJjaF1VpNhLRfIA4E7CFPirmz
[14]: https://docs.rs/stateright/
[15]: https://discord.gg/JbxGSVP4A6
[16]: https://github.com/stateright/stateright/tree/master/examples
[17]: https://github.com/stateright/stateright/blob/master/examples/paxos.rs
[18]: https://github.com/stateright/stateright/blob/master/examples/2pc.rs
[19]: https://en.wikipedia.org/wiki/Linearizability
[20]: https://en.wikipedia.org/wiki/Sequential_consistency
[21]: https://lamport.azurewebsites.net/tla/auxiliary/auxiliary.html
[22]: https://en.wikipedia.org/wiki/Formal_verification
[23]: https://lamport.azurewebsites.net/tla/tla.html
[24]: https://github.com/stateright/stateright/fork
[25]: https://help.github.com/articles/creating-a-pull-request-from-a-fork/
[26]: https://github.com/stateright/stateright/graphs/contributors
[27]: https://knockoutjs.com/
