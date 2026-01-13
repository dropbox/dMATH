## [About Jepsen][1]

Jepsen aims to improve the safety of distributed databases, queues, consensus systems, etc. We
maintain an [open source library][2] for safety testing, and publish free, [in-depth analyses][3] of
specific systems. In each analysis we explore whether the system lives up to its documentation’s
claims, file new bugs, and suggest recommendations for operators. In addition to [paid analysis][4],
Jepsen offers [technical talks][5], [training classes][6], and [consulting][7] services.

Jepsen pushes vendors to make accurate claims and test their software rigorously, helps users choose
databases and queues that fit their needs, and teaches engineers how to evaluate distributed systems
correctness for themselves.

## [See Also][8]

* [Github][9]
* [Mastodon][10]
* [Bluesky][11]
* [Threads][12]
* [LinkedIn][13]
* [Announcements mailing list][14]
* [General discussion mailing list][15]

## [News][16]

*Recent research, analyses, and announcements.*

### [NATS 2.12.1][17]

2025-12-07

[NATS][18] is a popular distributed streaming system. Jepsen tested NATS 2.12.1, focusing on its
durable JetStream subsystem, and found that it could [lose data or get stuck in persistent
split-brain][19] in response to file corruption or simulated node failures. This data loss was
caused in part by a default `fsync` policy which flushed data to disk once every two minutes, rather
than before acknowledgement. Even a single kernel crash or power failure, combined with process
pauses or network partitions, could cause NATS replicas to lose acknowledged messages. NATS has
documented its default lazy `fsync` setting, and is considering the other issues we found.

### [Jepsen 0.3.10][20]

2025-12-01

A new Jepsen release, [0.3.10][21], is now available on GitHub and Clojars. This release is aimed at
controllable entropy and support for running Jepsen inside Antithesis: a deterministic simulation
testing environment. A new supporting library, `jepsen.generator`, provides the current generator
system along with `jepsen.random`: a new namespace for pluggable random value generation. Jepsen
uses these RNGs throughout, which makes it possible to run a test with a deterministic seed, or to
source entropy from an external system, like Antithesis. The `jepsen.antithesis` library provides
additional support for assertions, randomness, and lifecycle operations, plus wrappers for clients
and checkers.

Also, this release introduces a new kind of visualization: op color plots, which show operations
over time with different user-defined colors. This is particularly helpful for getting a feeling for
“when did we lose data?” or “did only read-only queries succeed during a partition?”

### [A Distributed Systems Reliability Glossary][22]

2025-10-19

Jepsen and [Antithesis][23] wrote [A Distributed Systems Reliability Glossary][24]: a free reference
for engineers who build, test, and operate distributed systems. It covers basic concurrency theory,
consistency models, various faults, approaches to testing, and offers some links to further reading.

### [Jepsen 18: Serializable Mom][25]

2025-08-17

The latest Jepsen talk, “Jepsen 18: Serializable Mom”, is [now available on Youtube][26]. This talk
was presented on June 20, 2025, at [Systems Distributed][27] in Amsterdam. It covers [Bufstream
0.1.0][28], [Amazon RDS for PostgreSQL 17.4][29], and [TigerBeetle 0.16.1][30].

### [Capela dda5892][31]

2025-08-06

Jepsen and Capela, Inc worked together to test early builds of Capela, an unreleased distributed
programming environment. [Our analysis found twenty-two issues][32], including four problems in
Capela’s programming language semantics, fourteen crashes or non-fatal panics, severe performance
degradation after roughly a minute of operation, and three safety issues: partitions ignoring their
initial values, sporadically vanishing, and losing committed writes. Capela has fixed two of the
language issues—the others remain under investigation.

[All news from Jepsen…][33]

[1]: #about-jepsen
[2]: https://github.com/aphyr/jepsen
[3]: /analyses
[4]: https://jepsen.io/services/analysis
[5]: /talks
[6]: /services/training
[7]: /services/consulting
[8]: #see-also
[9]: https://github.com/jepsen-io
[10]: https://mastodon.jepsen.io/@jepsen
[11]: https://bsky.app/profile/jepsen.mastodon.jepsen.io.ap.brid.gy
[12]: https://www.threads.net/fediverse_profile/jepsen@mastodon.jepsen.io
[13]: https://www.linkedin.com/in/kyle-kingsbury/
[14]: https://groups.google.com/a/jepsen.io/forum/#!forum/announce
[15]: https://groups.google.com/a/jepsen.io/forum/#!forum/talk
[16]: /blog
[17]: /blog/2025-12-08-nats-2.12.1
[18]: https://nats.io
[19]: https://jepsen.io/analyses/nats-2.12.1
[20]: /blog/2025-12-02-jepsen-0.3.10
[21]: https://github.com/jepsen-io/jepsen/releases/tag/v0.3.10
[22]: /blog/2025-10-20-distsys-glossary
[23]: https://antithesis.com
[24]: https://antithesis.com/resources/reliability_glossary/
[25]: /blog/2025-08-18-jepsen-18-serializable-mom
[26]: https://www.youtube.com/watch?v=dpTxWePmW5Y
[27]: https://systemsdistributed.com/
[28]: https://jepsen.io/analyses/bufstream-0.1.0
[29]: http://jepsen.io/analyses/amazon-rds-for-postgresql-17.4
[30]: http://jepsen.io/analyses/tigerbeetle-0.16.11
[31]: /blog/2025-08-07-capela-dda5892
[32]: https://jepsen.io/analyses/capela-dda5892
[33]: /blog
