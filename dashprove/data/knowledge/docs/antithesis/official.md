# Documentation

### [Welcome to Antithesis][1]

Antithesis is an autonomous testing platform that finds the bugs in your software, with perfect
reproducibility to help you fix them. It supplements or replaces your existing testing tools and
lives alongside your normal CI workflow.

Our platform integrates [property-based testing][2], [fuzzing][3], and [deterministic simulation][4]
into a single testing tool.

Combining these approaches offers three key benefits:

1. You can test better by spending more compute time, not more developer time.
2. It finds bugs that you don’t know you have to look for.
3. It finds really hard bugs.

### [How to use Antithesis][5]

1. The first thing you’ll do is [upload your software][6] to a private container registry.
2. You’ll also upload a [test template][7] to exercise it.
3. From here, your tests will run in our [deterministic testing environment][8]. Kick off a test
   with a [webhook][9], or automatically with our [GitHub integration][10].
4. Explore your test results from our web UI, or receive them by email when your test run completes.
5. For tricky issues that require additional investigation, you can access a deterministic,
   time-traveling terminal session with our [multiverse debugger][11].

### [Getting help][12]

If you run into trouble, or have a particularly complex system to bring under test, or simply want
to make sure your testing is as thorough as possible, our customer success engineers would be happy
to help – reach out to us on [our Discord][13], or email [support@antithesis.com][14].

We’re confident that however you integrate Antithesis into your DevOps cycle, it will empower your
team to move faster and *build fearlessly,* knowing that you’ll find and fix issues before they
actually become problems.

### [What next?][15]

To learn more about how Antithesis works, [start here][16], or just follow the table of contents.

If you’re ready to get started, [contact us][17] to request a container registry and credentials.
Once you have those, you can go through our [tutorials][18] or dive right in with our [Docker Setup
Guide][19] or [Kubernetes Setup Guide][20]

[How Antithesis works][21]

[1]: #welcome-to-antithesis
[2]: /resources/property_based_testing/
[3]: /resources/reliability_glossary/#fuzz-testing
[4]: /resources/deterministic_simulation_testing/
[5]: #how-to-use-antithesis
[6]: /docs/getting_started/setup
[7]: /docs/test_templates/
[8]: /docs/environment/the_antithesis_environment/
[9]: /docs/webhook/test_webhook/
[10]: /docs/using_antithesis/ci/?#github-actions
[11]: /docs/multiverse_debugging/overview/
[12]: #getting-help
[13]: https://discord.com/invite/antithesis
[14]: mailto:support@antithesis.com
[15]: #what-next%3F
[16]: /docs/introduction/how_antithesis_works/
[17]: /contact/
[18]: /docs/tutorials/
[19]: /docs/getting_started/setup/
[20]: /docs/getting_started/setup_k8s/
[21]: /docs/introduction/how_antithesis_works/
