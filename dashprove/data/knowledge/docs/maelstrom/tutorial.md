We’ve teamed up with [Kyle Kingsbury][1], author of [Jepsen][2], to build this series of distributed
systems challenges so you can try your hand and see how your skills stack up.

The challenges are built on top of a platform called [Maelstrom][3], which in turn, is built on
Jepsen. This platform lets you build out a “node” in your distributed system and Maelstrom will
handle the routing of messages between the those nodes. This lets Maelstrom inject failures and
perform verification checks based on the consistency guarantees required by each challenge.

The documentation for these challenges will be in [Go][4], however, Maelstrom is language agnostic
so you can rework these challenges in any programming language.

## Got Stuck? Need Help?

It’s no secret that distributed systems are infuriating and difficult. Even the best developers in
the world can be brought to their knees in the face of cluster failures. If you get stuck on these
challenges or want to see how other folks are solving them, checkout the [#dist-sys-challenge][5]
tag on the [Fly.io Community Discourse][6].

## Let’s get started

Can’t wait to start? The [Echo challenge][7] will get you up and running with a basic echo
request/response to help you understand how Maelstrom works and to make sure you have everything
running correctly.

[1]: https://aphyr.com/about
[2]: https://jepsen.io/
[3]: https://github.com/jepsen-io/maelstrom
[4]: https://golang.org/
[5]: https://community.fly.io/tag/dist-sys-challenge
[6]: https://community.fly.io/
[7]: /dist-sys/1
