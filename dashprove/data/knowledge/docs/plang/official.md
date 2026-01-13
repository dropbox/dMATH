# Home

## Formal Modeling and Analysis of Distributed Systems

[[NuGet]][1] [[GitHub license]][2] [GitHub Action (CI on Windows)] [GitHub Action (CI on Ubuntu)]
[GitHub Action (CI on MacOS)]

**Challenge**: Distributed systems are notoriously hard to get right. Programming these systems is
challenging because of the need to reason about correctness in the presence of myriad possible
interleaving of messages and failures. Unsurprisingly, it is common for service teams to uncover
correctness bugs after deployment. Formal methods can play an important role in addressing this
challenge!

**P Overview:** P is a state machine based programming language for formally modeling and specifying
complex distributed systems. P allows programmers to model their system design as a collection of
communicating state machines. P supports several backend analysis engines (based on automated
reasoning techniques like model checking and symbolic execution) to check that the distributed
system modeled in P satisfy the desired correctness specifications.

> If you are wondering **"why do formal methods at all?"** or **"how is AWS using P to gain
> confidence in correctness of their services?"**, the following re:Invent 2023 talk answers this
> question, provides an overview of P, and its impact inside AWS: [(Re:Invent 2023 Talk) Gain
> confidence in system correctness & resilience with Formal Methods (Finding Critical Bugs
> Early!!)][3]

**Impact**: P is currently being used extensively inside Amazon (AWS) for analysis of complex
distributed systems. For example, Amazon S3 used P to formally reason about the core distributed
protocols involved in its strong consistency launch. Teams across AWS are now using P for thinking
and reasoning about their systems formally. P is also being used for programming safe robotics
systems in Academia. P was first used to implement and validate the USB device driver stack that
ships with Microsoft Windows 8 and Windows Phone.

**Experience and lessons learned**: In our experience of using P inside AWS, Academia, and
Microsoft. We have observed that P has helped developers in three critical ways: (1) **P as a
thinking tool**: Writing formal specifications in P forces developers to think about their system
design rigorously, and in turn helped in bridging gaps in their understanding of the system. A large
fraction of the bugs can be eliminated in the process of writing specifications itself! (2) **P as a
bug finder**: Model checking helped find corner case bugs in system design that were missed by
stress and integration testing. (3) **P helped boost developer velocity**: After the initial
overhead of creating the formal models, future updates and feature additions could be rolled out
faster as these non-trivial changes are rigorously validated before implementation.

[✨] ***Programming concurrent, distributed systems is fun but challenging, however, a pinch of
programming language design with a dash of automated reasoning can go a long way in addressing the
challenge and amplifying the fun!.*** [✨]

## Let the fun begin!

You can find most of the information about the P framework on this webpage: [what is P?][4],
[getting started][5], [tutorials][6], [case studies][7] and related [research publications][8]. If
you have any further questions, please feel free to create an [issue][9], ask on [discussions][10],
or [email us][11]

Contributions

*P has always been a collaborative project between industry and academia (since 2013) [🥁]. The P
team welcomes contributions and suggestions from all of you!! [👊].*

[1]: https://www.nuget.org/packages/P/
[2]: https://raw.githubusercontent.com/p-org/P/master/LICENSE.txt
[3]: https://youtu.be/FdXZXnkMDxs?si=iFqpl16ONKZuS4C0
[4]: whatisP/
[5]: getstarted/install/
[6]: tutsoutline/
[7]: casestudies/
[8]: publications/
[9]: https://github.com/p-org/P/issues
[10]: https://github.com/p-org/P/discussions
[11]: mailto:ankushdesai@gmail.com
