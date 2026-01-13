[[FsCheck logo - an inverted A, the mathematical symbol for forall.]][1]
** **

* FsCheck
* [Get via NuGet][2]
* [Source Code on GitHub][3]
* [License (BSD-3-Clause)][4]
* [Release Notes][5]
* [Who is using FsCheck?][6]
* Getting started
* [Quick Start][7]
* [Learning Resources][8]
* Documentation
* [Properties][9]
* [Generating test data][10]
* [Running tests][11]
* [Tips and Tricks][12]
* [Model-based testing vNext (Experimental)][13]
* API Reference
* [ All Namespaces ][14]

### [FsCheck][15]

# [FsCheck: Random Testing for .NET][16]

FsCheck is a tool for testing .NET programs automatically. The programmer provides a specification
of the program, in the form of properties which functions, methods or objects should satisfy, and
FsCheck then tests that the properties hold in a large number of randomly generated cases. While
writing the properties, you are actually writing a testable specification of your program.
Specifications are expressed in F#, C# or VB, using combinators defined in the FsCheck library.
FsCheck provides combinators to define properties, observe the distribution of test data, and define
test data generators. When a property fails, FsCheck automatically displays a minimal counter
example.

FsCheck, NUnit and xUnit.NET plugin can be [installed from NuGet][17] using your favorite package
manager.

Users have also created integrations with [Expecto][18], [Fuchu][19], [AutoFixture][20] and
[MSTest][21] that you can [find on NuGet][22].

## [Warning: Documentation][23]

The Documentation section below was written for version 2.x, but 3.x is the version that is
currently maintained. Only v3 will get new features and fixes, even though it's in pre-release.

This leaves us in the unfortunate position that some documentation is out of date and incomplete.
One exception is the API docs. They are generated from the 3.x code and are accurate - if in doubt,
believe the API docs.

The documentation and API docs for 2.x are not easily accessible anymore. The last commit of the 2.x
documentation site is [here][24].

Please help fixing this! FsCheck welcoms contributions of all kinds, big or small. See [issues][25]
for inspiration. Feel free to open an issue to highlight specific documentation problems or gaps,
even if you're not sure it really is a problem. At worst you'll get an answer to your question.

## [Documentation][26]

* [QuickStart][27] to get started.
* [Properties][28] describes FsCheck's language to express tests - in other frameworks these are
  often called parametrized tests or generative tests. FsCheck calls them properties.
* [Generating test data][29] describes how to guide FsCheck to generate better data or to stop it
  from generating data that doesn't make sense for what you're trying to test. FsCheck has a
  flexible language to describe test value generators and shrinkers, and apply them to your
  properties.
* [Running Tests][30] explains various ways to run FsCheck tests and how to integrate with unit
  testing frameworks.
* [Model based testing][31], for testing stateful systems and objects. Since this is in the
  Experimental namespace, semantic versioning promises do not apply to this part of the API.
* [Tips and tricks][32]
* [API Reference][33] contains automatically generated documentation for all types, modules and
  functions.

## [Contributing and copyright][34]

The project is hosted on [GitHub][35] where you can [report issues][36], fork the project and submit
pull requests. If you're adding new public API, please also consider adding [samples][37] that can
be turned into documentation.

The library is available under the BSD license, which allows modification and redistribution for
both commercial and non-commercial purposes. For more information see the [License file][38] in the
GitHub repository.

[1]: https://fscheck.github.io/FsCheck/
[2]: http://nuget.org/packages/FsCheck
[3]: https://github.com/fscheck/FsCheck
[4]: https://github.com/fscheck/FsCheck/blob/master/License.txt
[5]: https://github.com/fscheck/FsCheck/blob/master/FsCheck Release Notes.md
[6]: https://fscheck.github.io/FsCheck//users.html
[7]: https://fscheck.github.io/FsCheck//QuickStart.html
[8]: https://fscheck.github.io/FsCheck//LearningResources.html
[9]: https://fscheck.github.io/FsCheck//Properties.html
[10]: https://fscheck.github.io/FsCheck//TestData.html
[11]: https://fscheck.github.io/FsCheck//RunningTests.html
[12]: https://fscheck.github.io/FsCheck//TipsAndTricks.html
[13]: https://fscheck.github.io/FsCheck//StatefulTestingNew.html
[14]: https://fscheck.github.io/FsCheck/reference/index.html
[15]: https://fscheck.github.io/FsCheck/
[16]: #FsCheck-Random-Testing-for-NET
[17]: https://www.nuget.org/packages?q=fscheck
[18]: https://github.com/haf/Expecto
[19]: https://github.com/mausch/Fuchu
[20]: https://github.com/AutoFixture/AutoFixture
[21]: https://github.com/microsoft/testfx
[22]: https://www.nuget.org/packages?q=fscheck
[23]: #Warning-Documentation
[24]: https://github.com/fscheck/FsCheck/tree/1458b268b4311f7e4b25871715f1f9b5d58a21b3
[25]: https://github.com/fscheck/FsCheck/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22
[26]: #Documentation
[27]: QuickStart.html
[28]: Properties.html
[29]: TestData.html
[30]: RunningTests.html
[31]: StatefulTestingNew.html
[32]: TipsAndTricks.html
[33]: reference/index.html
[34]: #Contributing-and-copyright
[35]: https://github.com/fscheck/FsCheck
[36]: https://github.com/fscheck/FsCheck/issues
[37]: https://github.com/fscheck/FsCheck/tree/master/docs
[38]: https://github.com/fscheck/FsCheck/blob/master/License.txt
