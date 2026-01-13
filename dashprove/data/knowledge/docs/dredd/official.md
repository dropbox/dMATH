* »
* Dredd — HTTP API Testing Framework
* [ Edit on GitHub][1]

# Dredd — HTTP API Testing Framework[][2]

[[npm version]][3] [[Build Status]][4] [[Windows Build Status]][5] [[Documentation Status]][6]
[[Coverage Status]][7] [[Known Vulnerabilities]][8]

[Dredd - HTTP API Testing Framework]

> **Dredd is a language-agnostic command-line tool for validating API description document against
> backend implementation of the API.**

Dredd reads your API description and step by step validates whether your API implementation replies
with responses as they are described in the documentation.

## Features[][9]

### Supported API Description Formats[][10]

* [API Blueprint][11]
* [OpenAPI 2][12] (formerly known as Swagger)
* [OpenAPI 3][13] ([experimental][14], contributions welcome!)

### Supported Hooks Languages[][15]

Dredd supports writing [hooks][16] — a glue code for each test setup and teardown. Following
languages are supported:

* [Go][17]
* [Node.js (JavaScript)][18]
* [Perl][19]
* [PHP][20]
* [Python][21]
* [Ruby][22]
* [Rust][23]
* Didn’t find your favorite language? [Add a new one!][24]

### Supported Systems[][25]

* Linux, macOS, Windows, …
* [Travis CI][26], [CircleCI][27], [Jenkins][28], [AppVeyor][29], …

## Contents[][30]

* [Installation][31]
  
  * [Docker][32]
  * [npm][33]
* [Quickstart][34]
  
  * [Install Dredd][35]
  * [Document Your API][36]
  * [Implement Your API][37]
  * [Test Your API][38]
  * [Configure Dredd][39]
  * [Use Hooks][40]
  * [Advanced Examples][41]
* [How It Works][42]
  
  * [Versioning][43]
  * [Execution Life Cycle][44]
  * [Automatic Expectations][45]
  * [Making Your API Description Ready for Testing][46]
  * [Choosing HTTP Transactions][47]
  * [Security][48]
  * [Using HTTP(S) Proxy][49]
* [How-To Guides][50]
  
  * [Isolation of HTTP Transactions][51]
  * [Testing API Workflows][52]
  * [Making Dredd Validation Stricter][53]
  * [Integrating Dredd with Your Test Suite][54]
  * [Continuous Integration][55]
  * [Authenticated APIs][56]
  * [Sending Multipart Requests][57]
  * [Sending Form Data][58]
  * [Working with Images and other Binary Bodies][59]
  * [Multiple Requests and Responses][60]
  * [Using Apiary Reporter and Apiary Tests][61]
  * [Example Values for Request Parameters][62]
  * [Removing Sensitive Data from Test Reports][63]
* [Usage][64]
  
  * [Usage][65]
  * [Arguments][66]
  * [Configuration File][67]
  * [CLI Options Reference][68]
* [JavaScript API][69]
  
  * [Configuration Object for Dredd Class][70]
* [Hooks][71]
  
  * [Getting started][72]
  * [Supported languages][73]
  * [Transaction names][74]
  * [Types of hooks][75]
  * [Hooks inside Docker][76]
* [Data Structures][77]
  
  * [Transaction (object)][78]
  * [Transaction Test (object)][79]
  * [Transaction Results (object)][80]
  * [Gavel Validation Result Field (object)][81]
  * [Gavel Error (object)][82]
  * [Test Runtime Error (object)][83]
  * [Apiary Reporter Test Data (object)][84]
  * [Internal Apiary Data Structures][85]
* [Internals][86]
  
  * [Maintainers][87]
  * [Contributing][88]
  * [Contributing to documentation][89]
  * [Windows support][90]
  * [API description parsing][91]
  * [Architecture][92]

## Useful Links[][93]

* [GitHub Repository][94]
* [Bug Tracker][95]
* [Changelog][96]

## Example Applications[][97]

* [Express.js][98]
* [Laravel][99]
* [Laravel & OpenAPI 3][100]
* [Ruby on Rails][101]
[Next ][102]

Revision `3d4ae143`.

Built with [Sphinx][103] using a [theme][104] provided by [Read the Docs][105].

[1]: https://github.com/apiaryio/dredd/blob/master/docs/index.rst
[2]: #dredd-http-api-testing-framework
[3]: https://www.npmjs.com/package/dredd
[4]: https://circleci.com/gh/apiaryio/dredd/tree/master
[5]: https://ci.appveyor.com/project/Apiary/dredd/branch/master
[6]: https://readthedocs.org/projects/dredd/builds/
[7]: https://coveralls.io/github/apiaryio/dredd
[8]: https://snyk.io/test/npm/dredd
[9]: #features
[10]: #supported-api-description-formats
[11]: https://apiblueprint.org
[12]: https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md
[13]: https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md
[14]: https://github.com/apiaryio/api-elements.js/blob/master/packages/openapi3-parser/STATUS.md
[15]: #supported-hooks-languages
[16]: hooks/index.html#hooks
[17]: hooks/go.html#hooks-go
[18]: hooks/js.html#hooks-nodejs
[19]: hooks/perl.html#hooks-perl
[20]: hooks/php.html#hooks-php
[21]: hooks/python.html#hooks-python
[22]: hooks/ruby.html#hooks-ruby
[23]: hooks/rust.html#hooks-rust
[24]: hooks/new-language.html#hooks-new-language
[25]: #supported-systems
[26]: https://travis-ci.org
[27]: https://circleci.com
[28]: https://jenkins.io
[29]: https://www.appveyor.com
[30]: #contents
[31]: installation.html
[32]: installation.html#install-docker
[33]: installation.html#npm
[34]: quickstart.html
[35]: quickstart.html#install-dredd
[36]: quickstart.html#document-your-api
[37]: quickstart.html#implement-your-api
[38]: quickstart.html#test-your-api
[39]: quickstart.html#configure-dredd
[40]: quickstart.html#use-hooks
[41]: quickstart.html#advanced-examples
[42]: how-it-works.html
[43]: how-it-works.html#versioning
[44]: how-it-works.html#execution-life-cycle
[45]: how-it-works.html#automatic-expectations
[46]: how-it-works.html#making-your-api-description-ready-for-testing
[47]: how-it-works.html#choosing-http-transactions
[48]: how-it-works.html#security
[49]: how-it-works.html#using-https-proxy
[50]: how-to-guides.html
[51]: how-to-guides.html#isolation-of-http-transactions
[52]: how-to-guides.html#testing-api-workflows
[53]: how-to-guides.html#making-dredd-validation-stricter
[54]: how-to-guides.html#integrating-dredd-with-your-test-suite
[55]: how-to-guides.html#continuous-integration
[56]: how-to-guides.html#authenticated-apis
[57]: how-to-guides.html#sending-multipart-requests
[58]: how-to-guides.html#sending-form-data
[59]: how-to-guides.html#working-with-images-and-other-binary-bodies
[60]: how-to-guides.html#multiple-requests-and-responses
[61]: how-to-guides.html#using-apiary-reporter-and-apiary-tests
[62]: how-to-guides.html#example-values-for-request-parameters
[63]: how-to-guides.html#removing-sensitive-data-from-test-reports
[64]: usage-cli.html
[65]: usage-cli.html#usage
[66]: usage-cli.html#arguments
[67]: usage-cli.html#configuration-file
[68]: usage-cli.html#cli-options-reference
[69]: usage-js.html
[70]: usage-js.html#configuration-object-for-dredd-class
[71]: hooks/index.html
[72]: hooks/index.html#getting-started
[73]: hooks/index.html#supported-languages
[74]: hooks/index.html#getting-transaction-names
[75]: hooks/index.html#types-of-hooks
[76]: hooks/index.html#hooks-inside-docker
[77]: data-structures.html
[78]: data-structures.html#transaction-object
[79]: data-structures.html#transaction-test-object
[80]: data-structures.html#transaction-results-object
[81]: data-structures.html#gavel-validation-result-field-object
[82]: data-structures.html#gavel-error-object
[83]: data-structures.html#test-runtime-error-object
[84]: data-structures.html#apiary-reporter-test-data-object
[85]: data-structures.html#internal-apiary-data-structures
[86]: internals.html
[87]: internals.html#maintainers
[88]: internals.html#contributing
[89]: internals.html#contributing-to-documentation
[90]: internals.html#windows-support
[91]: internals.html#api-description-parsing
[92]: internals.html#architecture
[93]: #useful-links
[94]: https://github.com/apiaryio/dredd
[95]: https://github.com/apiaryio/dredd/issues?q=is%3Aopen
[96]: https://github.com/apiaryio/dredd/releases
[97]: #example-applications
[98]: https://github.com/apiaryio/dredd-example
[99]: https://github.com/ddelnano/dredd-hooks-php/wiki/Laravel-Example
[100]: https://github.com/AndyWendt/laravel-dredd-openapi-v3
[101]: https://gitlab.com/theodorton/dredd-test-rails/
[102]: installation.html
[103]: https://www.sphinx-doc.org/
[104]: https://github.com/readthedocs/sphinx_rtd_theme
[105]: https://readthedocs.org
