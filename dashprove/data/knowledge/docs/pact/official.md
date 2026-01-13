* Getting started
* Introduction
On this page

# Introduction

Pact is a code-first tool for testing HTTP and message integrations using `contract tests`. Contract
tests assert that inter-application messages conform to a shared understanding that is documented in
a contract. Without contract testing, the only way to ensure that applications will work correctly
together is by using expensive and brittle integration tests.

Do you [set your house on fire to test your smoke alarm?][1] No, you test the contract it holds with
your ears by using the testing button. Pact provides that testing button for your code, allowing you
to safely confirm that your applications will work together without having to deploy the world
first.

To view an animated step-by-step explanation of how Pact works, check out this [How Pact works][2]
(external↗️) page.

[[How Pact works preview]][3]

## Watch a video[​][4]

Or, watch the [full series on contract testing][5].

## Ready to jump into the code already?[​][6]

Get started with our [5 minute guide][7].

## What is contract testing?[​][8]

***Contract testing** is a technique for testing an integration point by checking each application
in isolation to ensure the messages it sends or receives conform to a shared understanding that is
documented in a "contract".*

For applications that communicate via HTTP, these "messages" would be the HTTP request and response,
and for an application that used queues, this would be the message that goes on the queue.

In practice, a common way of implementing contract tests (and the way Pact does it) is to check that
all the calls to your test doubles [return the same results][9] as a call to the real application
would.

## When would I use contract testing?[​][10]

Contract testing is immediately applicable anywhere where you have two services that need to
communicate - such as an API client and a web front-end. Although a single client and a single
service is a common use case, contract testing really shines in an environment with many services
(as is common for a microservice architecture). Having well-formed contract tests makes it easy for
developers to avoid version hell. Contract testing is the killer app for microservice development
and deployment.

## Contract testing terminology[​][11]

In general, a contract is between a *consumer* (for example, a client that wants to receive some
data) and a *provider* (for example, an API on a server that provides the data the client needs). In
microservice architectures, the traditional terms *client* and *server* are not always appropriate
-- for example, when communication is achieved through message queues. For this reason, we stick to
*consumer* and *provider* in this documentation.

## Consumer Driven Contracts[​][12]

Pact is a code-first [*consumer-driven*][13] contract testing tool, and is generally used by
developers and testers who code. The contract is generated during the execution of the automated
consumer tests. A major advantage of this pattern is that only parts of the communication that are
actually used by the consumer(s) get tested. This in turn means that any provider behaviour not used
by current consumers is free to change without breaking tests.

Unlike a schema or specification (eg. OAS), which is a static artefact that describes all possible
states of a resource, a Pact contract is enforced by executing a collection of test cases, each of
which describes a single concrete request/response pair - Pact is, in effect, "contract by example".
Read more on the [difference between schema testing and contract testing][14].

## Provider contract testing[​][15]

The term "contract testing", or "provider contract testing", is sometimes used in other literature
and documentation in the context of a standalone provider application (rather than in the context of
an integration). When used in this context, "contract testing" means: a technique for ensuring a
provider's actual behaviour conforms to its documented contract (for example, an OpenAPI document).
This type of contract testing helps avoid integration failures by ensuring the provider code and
documentation are in sync with each other. On its own, however, it does not provide any test based
assurance that the consumers are calling the provider in the correct manner, or that the provider
can meet all its consumers' expectations, and hence, it is not as effective in preventing
integration bugs.

Whenever the Pact documentation references "contract testing" it is referring to "integration
contract testing" as described previously in this page.

[Edit this page][16]
Last updated on Aug 30, 2022 by Matt Fellows

[1]: https://dius.com.au/2014/05/20/simplifying-microservice-testing-with-pacts/
[2]: https://pactflow.io/how-pact-works?utm_source=ossdocs&utm_campaign=getting_started
[3]: https://pactflow.io/how-pact-works?utm_source=ossdocs&utm_campaign=getting_started
[4]: #watch-a-video
[5]: https://www.youtube.com/embed/videoseries?list=PLwy9Bnco-IpfZ72VQ7hce8GicVZs7nm0i
[6]: #ready-to-jump-into-the-code-already
[7]: /5-minute-getting-started-guide
[8]: #what-is-contract-testing
[9]: https://martinfowler.com/bliki/ContractTest.html
[10]: #when-would-i-use-contract-testing
[11]: #contract-testing-terminology
[12]: #consumer-driven-contracts
[13]: https://martinfowler.com/articles/consumerDrivenContracts.html
[14]: https://pactflow.io/blog/contract-testing-using-json-schemas-and-open-api-part-1/
[15]: #provider-contract-testing
[16]: https://github.com/pact-foundation/docs.pact.io/edit/master/website/docs/getting_started.md
