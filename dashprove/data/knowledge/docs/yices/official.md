─────────┬─┬─────────┬─┬─────────────┬─┬─────────────
[Home][1]│•│[Docs][2]│•│[Contacts][3]│•│[FM Tools][4]
─────────┴─┴─────────┴─┴─────────────┴─┴─────────────

[[Logo]][5]

### Download

* [Source Code][6]
* [Linux (64 bits)][7]
* [Mac OS X (64 bits Intel)][8]
* [Mac OS X (64 bits Apple-Silicon)][9]
* [Windows (64 bits)][10]

### Documentation

* [Manual][11]
* [API Reference][12]
* [Publications][13]

### Support

* [FAQ][14]
* [Get Help][15]
* [Report a Bug][16]

# The Yices SMT Solver

Yices 2 is an [SMT][17] solver that decides the satisfiability of formulas containing uninterpreted
function symbols with equality, real and integer arithmetic, bitvectors, scalar types, and tuples.
Yices 2 supports both linear and nonlinear arithmetic.

Yices 2 can process input written in the [SMT-LIB notation][18] (both versions 2.0 and 1.2 are
supported). Alternatively, you can write specifications using Yices 2's own specification language,
which includes tuples and scalar types. You can also use Yices 2 as a library in your software.

Yices is open source software distributed under the [GPLv3 license][19]. The Yices source is
available on [GitHub][20]. To discuss alternative license terms, please contact
[fm-licensing@csl.sri.com][21].

## Downloads

You can install Yices with homebrew on Mac OS X or with apt or aptitude on Debian/Ubuntu Linux. We
also kindly provide precompiled binaries of the latest stable release below.

On Ubuntu or Debian:

sudo add-apt-repository ppa:sri-csl/formal-methods
sudo apt-get update
sudo apt-get install yices2

On Mac OS X:

brew install SRI-CSL/sri-csl/yices2

### Latest Release

The current version is Yices 2.7.0. It was released on 2025-07-16.

* [Documentation][22]
* [Release notes][23]
* [Source Code][24]
* Binaries
  
  * Linux (64 bits)[ AMD/Intel][25]
  * Mac OS X (64 bits) [ Intel][26] [ Apple-Silicon][27]
  * Windows [ (64 bits)][28]

### Previous Releases

Older versions of Yices can be downloaded [here][29].

We no longer maintain Yices 1 but you can still download it [here][30] and the old documentation is
[here][31].

## Contact Us

* [Get Help][32]
* [Report a Bug][33]
* [Subscribe or Post to the Yices mailing lists][34]

## Acknowledgements

Yices 2 is developed by [SRI International][35]'s [Computer Science Laboratory][36]. The development
of Yices has been funded by SRI International, the National Science Foundation, the National
Aeronautics and Space Administration, and the Defense Advanced Research Projects Agency.

SRI International distributes other [formal method tools][37] that you may find useful.

─────────┬─┬─────────┬─┬─────────────┬─┬─────────────
[Home][38│•│[Docs][39│•│[Contacts][40│•│[FM          
]        │ │]        │ │]            │ │Tools][41]   
─────────┴─┴─────────┴─┴─────────────┴─┴─────────────

[1]: index.html
[2]: yices2-documentation.html
[3]: mailing_lists.html
[4]: http://fm.csl.sri.com
[5]: #
[6]: releases/2.7.0/yices-2.7.0-src.tar.gz
[7]: releases/2.7.0/yices-2.7.0-x86_64-pc-linux-gnu-static-gmp.tar.gz
[8]: releases/2.7.0/yices-2.7.0-x86_64-apple-darwin21.6.0-static-gmp.tar.gz
[9]: releases/2.7.0/yices-2.7.0-arm-apple-darwin22.6.0-static-gmp.tar.gz
[10]: releases/2.7.0/yices-2.7.0-x86_64-unknown-mingw32-static-gmp.zip
[11]: papers/manual.pdf
[12]: doc/index.html
[13]: yices2-documentation.html#publis
[14]: faq.html
[15]: help.html
[16]: bugs.html
[17]: http://en.wikipedia.org/wiki/Satisfiability_Modulo_Theories
[18]: http://www.smtlib.org/
[19]: https://www.gnu.org/licenses/gpl-3.0.en.html
[20]: https://github.com/SRI-CSL/yices2
[21]: mailto:fm-licensing@csl.sri.com
[22]: yices2-documentation.html
[23]: release-notes.html
[24]: releases/2.7.0/yices-2.7.0-src.tar.gz
[25]: releases/2.7.0/yices-2.7.0-x86_64-pc-linux-gnu-static-gmp.tar.gz
[26]: releases/2.7.0/yices-2.7.0-x86_64-apple-darwin21.6.0-static-gmp.tar.gz
[27]: releases/2.7.0/yices-2.7.0-arm-apple-darwin22.6.0-static-gmp.tar.gz
[28]: releases/2.7.0/yices-2.7.0-x86_64-unknown-mingw32-static-gmp.zip
[29]: download-old.html
[30]: old/download-yices1.html
[31]: old/yices1-documentation.html
[32]: help.html
[33]: bugs.html
[34]: mailing_lists.html
[35]: http://www.sri.com/
[36]: http://www.sri.com/about/organization/information-computing-sciences/computer-science-laborato
ry
[37]: http://fm.csl.sri.com/
[38]: index.html
[39]: yices2-documentation.html
[40]: mailing_lists.html
[41]: http://fm.csl.sri.com
