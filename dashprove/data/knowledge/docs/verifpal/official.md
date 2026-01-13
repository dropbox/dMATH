# Verifpal^{®}

Verifpal^{®} is new software for verifying the security of cryptographic protocols. Building upon
contemporary research in symbolic formal verification, Verifpal’s main aim is to appeal more to
real-world practitioners, students and engineers without sacrificing comprehensive formal
verification features.

In order to achieve this, Verifpal introduces a new, intuitive language for modeling protocols that
is much easier to write and understand than the languages employed by existing tools. At the same
time, Verifpal is able to model protocols under an active attacker with unbounded sessions and fresh
values, and supports queries for advanced security properties such as forward secrecy or key
compromise impersonation.

Verifpal has already been used to verify security properties for Signal, Scuttlebutt, TLS 1.3,
Telegram and other protocols. It is a community-focused project, and available under a GPLv3
license.

### An Intuitive Protocol Modeling Language

The Verifpal language is meant to illustrate protocols close to how one may describe them in an
informal conversation, while still being precise and expressive enough for formal modeling.

### Modeling that Avoids User Error

Verifpal does not allow users to define their own cryptographic primitives. Instead, it comes with
built-in cryptographic functions — this is meant to remove the potential for users to define
fundamental cryptographic operations incorrectly.

### Easy to Understand Analysis Output

When a contradiction is found for a query, the result is related in a readable format that ties the
attack to a real-world scenario. This is done by using terminology to indicate how the attack could
have been possible.

### Friendly and Integrated Software

Verifpal comes with a Visual Studio Code extension that offers syntax highlighting, live query
analysis, protocol diagram visualizations and more, allowing developers to obtain insights on their
model as they are writing it.

Check out [this cool three-minute demonstration of how Verifpal can be used to model a protocol with
properties such as forward secrecy on YouTube!][1]

[1]: https://www.youtube.com/watch?v=JrJt6hOAQlQ
