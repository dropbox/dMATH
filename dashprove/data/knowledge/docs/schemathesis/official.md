# Schemathesis

Schemathesis automatically generates property-based tests from your OpenAPI or GraphQL schema and
exercises the edge cases that break your API.

[Schemathesis automatically finding a server error]
*Finding a server error that manual testing missed*

[Find Bugs in Your API in 5 Minutes ][1]

## Try it now

`uvx schemathesis run https://example.schemathesis.io/openapi.json
`

This command will immediately find real bugs in a demo API and show you exactly how to reproduce
them.

## Why teams choose Schemathesis

* 🎯 Find real bugs on the first run (commonly 5–15 in production schemas).
* ⏱️ Point it at your schema—no per-endpoint tests to maintain.
* 🔄 Keeps pace with the spec; new endpoints are covered automatically.
* 🔌 Exports JUnit, HAR, and integrates with `pytest` or CI/CD.
* 📑 Uses your OpenAPI or GraphQL schema as the single source of truth.

Developer feedback

"The tool is amazing as it can test negative scenarios instead of me and much faster!"

*— Luděk Nový, JetBrains*

## Documentation

* **New to Schemathesis?**
  
  Get started in minutes:
  
  * [ Quick Start - 5 minutes][2]
  * [ CLI Tutorial - 20 minutes][3]
  * [ Pytest Tutorial - 15 minutes][4]
* **How-To Guides**
  
  Practical guides for common scenarios:
  
  * [ CI/CD Integration][5]
  * [ Extending Schemathesis][6]
  * [ More...][7]
* **Want to understand how it works?**
  
  Deep dive into concepts:
  
  * [ Data Generation][8]
  * [ Example Testing][9]
  * [ Stateful Testing][10]
* **Need technical details?**
  
  Complete reference:
  
  * [ Command-Line Interface][11]
  * [ Python API][12]
  * [ Configuration File][13]

Important

**Upgrading from older versions?** See the [Migration Guide][14] for key changes.

## Need help?

* **[Resources][15]** — Community articles, videos, and tutorials
* **[FAQ][16]** — Frequently Asked Questions

[1]: quick-start/
[2]: quick-start/
[3]: tutorials/cli/
[4]: tutorials/pytest/
[5]: guides/cicd/
[6]: guides/extending/
[7]: guides/
[8]: explanations/data-generation/
[9]: explanations/examples/
[10]: explanations/stateful/
[11]: reference/cli/
[12]: reference/python/
[13]: reference/configuration/
[14]: migration/
[15]: resources/
[16]: faq/
