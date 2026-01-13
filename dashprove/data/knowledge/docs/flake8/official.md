* Flake8: Your Tool For Style Guide Enforcement
* [ View page source][1]

# Flake8: Your Tool For Style Guide Enforcement[][2]

## Quickstart[][3]

### Installation[][4]

To install **Flake8**, open an interactive shell and run:

python<version> -m pip install flake8

If you want **Flake8** to be installed for your default Python installation, you can instead use:

python -m pip install flake8

Note

It is **very** important to install **Flake8** on the *correct* version of Python for your needs. If
you want **Flake8** to properly parse new language features in Python 3.5 (for example), you need it
to be installed on 3.5 for **Flake8** to understand those features. In many ways, Flake8 is tied to
the version of Python on which it runs.

### Using Flake8[][5]

To start using **Flake8**, open an interactive shell and run:

flake8 path/to/code/to/check.py
# or
flake8 path/to/code/

Note

If you have installed **Flake8** on a particular version of Python (or on several versions), it may
be best to instead run `python<version> -m flake8`.

If you only want to see the instances of a specific warning or error, you can *select* that error
like so:

flake8 --select E123,W503 path/to/code/

Alternatively, if you want to add a specific warning or error to *ignore*:

flake8 --extend-ignore E203,W234 path/to/code/

Please read our user guide for more information about how to use and configure **Flake8**.

## FAQ and Glossary[][6]

* [Frequently Asked Questions][7]
  
  * [When is Flake8 released?][8]
  * [How can I help Flake8 release faster?][9]
  * [What is the next version of Flake8?][10]
  * [Why does Flake8 use ranges for its dependencies?][11]
  * [Should I file an issue when a new version of a dependency is available?][12]
* [Glossary of Terms Used in Flake8 Documentation][13]

## User Guide[][14]

All users of **Flake8** should read this portion of the documentation. This provides examples and
documentation around **Flake8**’s assortment of options and how to specify them on the command-line
or in configuration files.

* [Using Flake8][15]
  
  * [Invoking Flake8][16]
  * [Configuring Flake8][17]
  * [Full Listing of Options and Their Descriptions][18]
  * [Error / Violation Codes][19]
  * [Selecting and Ignoring Violations][20]
  * [Using Plugins For Fun and Profit][21]
  * [Using Version Control Hooks][22]
  * [Public Python API][23]

## Plugin Developer Guide[][24]

If you’re maintaining a plugin for **Flake8** or creating a new one, you should read this section of
the documentation. It explains how you can write your plugins and distribute them to others.

* [Writing Plugins for Flake8][25]
  
  * [Getting Started][26]
  * [Video Tutorial][27]
  * [Detailed Plugin Development Documentation][28]

## Contributor Guide[][29]

If you are reading **Flake8**’s source code for fun or looking to contribute, you should read this
portion of the documentation. This is a mix of documenting the internal-only interfaces **Flake8**
and documenting reasoning for Flake8’s design.

* [Exploring Flake8’s Internals][30]
  
  * [Contributing to Flake8][31]
  * [Writing Documentation for Flake8][32]
  * [Writing Code for Flake8][33]
  * [Releasing Flake8][34]
  * [What Happens When You Run Flake8][35]
  * [How Checks are Run][36]
  * [Command Line Interface][37]
  * [Built-in Formatters][38]
  * [Option and Configuration Handling][39]
  * [Plugin Handling][40]
  * [Utility Functions][41]

## Release Notes and History[][42]

* [Release Notes and History][43]
  
  * [7.x Release Series][44]
  * [6.x Release Series][45]
  * [5.x Release Series][46]
  * [4.x Release Series][47]
  * [3.x Release Series][48]
  * [2.x Release Series][49]
  * [1.x Release Series][50]
  * [0.x Release Series][51]

## General Indices[][52]

* [Index][53]
* [Index of Documented Public Modules][54]
* [Glossary of terms][55]
[Next ][56]

© Copyright 2016, Ian Stapleton Cordasco.

Built with [Sphinx][57] using a [theme][58] provided by [Read the Docs][59].

[1]: _sources/index.rst.txt
[2]: #flake8-your-tool-for-style-guide-enforcement
[3]: #quickstart
[4]: #installation
[5]: #using-flake8
[6]: #faq-and-glossary
[7]: faq.html
[8]: faq.html#when-is-flake8-released
[9]: faq.html#how-can-i-help-flake8-release-faster
[10]: faq.html#what-is-the-next-version-of-flake8
[11]: faq.html#why-does-flake8-use-ranges-for-its-dependencies
[12]: faq.html#should-i-file-an-issue-when-a-new-version-of-a-dependency-is-available
[13]: glossary.html
[14]: #user-guide
[15]: user/index.html
[16]: user/invocation.html
[17]: user/configuration.html
[18]: user/options.html
[19]: user/error-codes.html
[20]: user/violations.html
[21]: user/using-plugins.html
[22]: user/using-hooks.html
[23]: user/python-api.html
[24]: #plugin-developer-guide
[25]: plugin-development/index.html
[26]: plugin-development/index.html#getting-started
[27]: plugin-development/index.html#video-tutorial
[28]: plugin-development/index.html#detailed-plugin-development-documentation
[29]: #contributor-guide
[30]: internal/index.html
[31]: internal/contributing.html
[32]: internal/writing-documentation.html
[33]: internal/writing-code.html
[34]: internal/releases.html
[35]: internal/start-to-finish.html
[36]: internal/checker.html
[37]: internal/cli.html
[38]: internal/formatters.html
[39]: internal/option_handling.html
[40]: internal/plugin_handling.html
[41]: internal/utils.html
[42]: #release-notes-and-history
[43]: release-notes/index.html
[44]: release-notes/index.html#x-release-series
[45]: release-notes/index.html#id1
[46]: release-notes/index.html#id2
[47]: release-notes/index.html#id3
[48]: release-notes/index.html#id4
[49]: release-notes/index.html#id5
[50]: release-notes/index.html#id6
[51]: release-notes/index.html#id7
[52]: #general-indices
[53]: genindex.html
[54]: py-modindex.html
[55]: glossary.html#glossary
[56]: faq.html
[57]: https://www.sphinx-doc.org/
[58]: https://github.com/readthedocs/sphinx_rtd_theme
[59]: https://readthedocs.org
