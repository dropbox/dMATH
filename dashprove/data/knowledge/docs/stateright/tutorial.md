> [links] [[chat]][1] [[crates.io]][2] [[docs.rs]][3] [[stars]][4]

# [Building Distributed Systems With Stateright][5]

*by Jonathan Nadal*

> Deep understanding of causality sometimes requires the understanding of very large patterns and
> their abstract relationships and interactions, not just the understanding of microscopic objects
> interacting in microscopic time intervals.
> 
> ― Douglas R. Hofstadter, *I Am a Strange Loop*

Distributed computing is a term that refers to multiple computers working together *as a system* to
solve a problem, typically because that problem would not be solvable on a single computer. For
example, we all want to know that our important files will be accessible even when computer hardware
inevitably fails. As a second example, a researcher at a pharmaceutical company may have a complex
problem that would take decades for a single computer to solve, but which a collection of computers
working together could solve in days.

Unique algorithms must be employed to coordinate workloads across these geographically distributed
systems of computers because they are susceptible to categories of nondeterminism that do not arise
when a problem is solved with a single computer. For example, the networks that link these computers
will drop, reorder, and even redeliver messages. Algorithms that fail to account for this behavior
may run correctly for extended periods but will eventually fail at unpredicatable times in
unpredictable ways, such as causing data corruption.

Stateright is a software framework for analyzing and systematically verifying distributed systems.
Its name refers to the goal of verifying that a system's collective state always satisfies a
correctness specification, such as "any data written to the system should be accessible as long as
at least one data center is reachable."

Cloud service providers like AWS and Azure leverage verification software such as [the TLA+ model
checker][6] to achieve the same goal, but whereas those solutions typically verify a high level
system design, Stateright is able to verify the underlying system *implementation* in addition to
the design (along with providing other unique benefits explained in the "[Comparison with TLA+][7]"
chapter). On the other end of the spectrum are tools such as [Jepsen][8] which can validate a final
implementation by testing a random subset of the system's behavior, whereas Stateright can
systematically enumerate all possible behaviors within a specified model of the system.

We'll jump right in with a motivating example in the first chapter, [Getting Started][9].
Alternatively:

* See the [Stateright YouTube channel][10] if you prefer a quick video introduction.
* For a broader overview of Stateright see the related Distributed Systems Reading Group [special
  session][11]. The [slides][12] are available on this site (`Esc` key zooms out, and note that
  there are `Up`/`Down` slides). Murat Demirbas also wrote an [associated blog post][13].

[1]: https://discord.gg/JbxGSVP4A6
[2]: https://crates.io/crates/stateright
[3]: https://docs.rs/stateright
[4]: https://github.com/stateright/stateright
[5]: #building-distributed-systems-with-stateright
[6]: https://lamport.azurewebsites.net/tla/industrial-use.html
[7]: ./comparison-with-tlaplus.html
[8]: https://jepsen.io/
[9]: ./getting-started.html
[10]: https://www.youtube.com/playlist?list=PLUhyBsVvEJjaF1VpNhLRfIA4E7CFPirmz
[11]: http://charap.co/building-distributed-systems-with-stateright/
[12]: https://www.stateright.rs/2021-03-30
[13]: https://muratbuffalo.blogspot.com/2021/04/building-distributed-systems-with.html
