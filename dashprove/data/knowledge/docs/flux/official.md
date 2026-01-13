[Flux Logo]

Flux is a [**refinement type checker**][1] plugin for Rust that lets you *specify* a range of
correctness properties and have them be *verified* at compile time.

Flux works by extending Rust's types with [refinements][2]: logical assertions describing additional
correctness requirements that are checked during compilation, thereby eliminating various classes of
run-time problems.

You can try it on the [**online playground**][3].

Better still, read the [**interactive tutorial**][4], to learn how you can use Flux on your Rust
code.

[1]: https://github.com/flux-rs/flux/
[2]: https://arxiv.org/abs/2010.07763
[3]: https://flux.goto.ucsd.edu/
[4]: ./tutorial/01-refinements.html
