# include-what-you-use

A tool for use with clang to analyze #includes in C and C++ source files

* [Discussion list][1]
* [Issues][2]
* [Source][3]
* **Docs**
  
  * [Instructions for developers][4]
  * [Instructions for users][5]
  * [Why include what you use][6]
  * [All docs][7]
* **Downloads**
  
  * [All downloads][8]
  * [Download clang][9]

"Include what you use" means this: for every symbol (type, function variable, or macro) that you use
in foo.cc, either foo.cc or foo.h should #include a .h file that exports the declaration of that
symbol. The include-what-you-use tool is a program that can be built with the clang libraries in
order to analyze #includes of source files to find include-what-you-use violations, and suggest
fixes for them.

The main goal of include-what-you-use is to remove superfluous #includes. It does this both by
figuring out what #includes are not actually needed for this file (for both .cc and .h files), and
replacing #includes with forward-declares when possible.

#### 20 Sep 2025

iwyu 0.25 compatible with llvm+clang 21 is released. Major changes:

* [iwyu] Add support for GNU `__cleanup__` attribute
* [iwyu] Improve handling of member pointers
* [iwyu] Improve reporting of C arrays
* [iwyu] Improve understanding of builtin type traits
* [iwyu] Add `--export_mappings` option to generate external mappings from internal (breaking
  change)
* [iwyu_tool] Add `--exclude` option to skip individual source files
* [mappings] Add mapping generator for Apple Libc
* [mappings] Improve libc and POSIX mappings
* [ci] Add a [test suite for all reproduced open bugs][10]

Note the breaking change for `.imp` external mappings; IWYU no longer ships `.imp` files with the
mappings already built into the `include-what-you-use` executable. Instead, use
`include-what-you-use --export_mappings` to write them out on demand. See [documentation][11] for
more information.

For the full list of closed issues see [the `iwyu 0.25` milestone][12].

Contributions in this release by Aaron Puchert, Alex Overchenko, Bolshakov, Daan De Meyer, Kim
Gräsman, Mohamed Akram. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.25.src.tar.gz][13]. It is equivalent
to the `0.25` tag and `clang_21` branch.

#### 05 Apr 2025

iwyu 0.24 compatible with llvm+clang 20 is released. Major changes:

* [iwyu] Improve handling of type traits
* [iwyu] Improve type analysis
* [mappings] Improve libstdc++ mappings
* [fix_includes] Implement `--quoted_includes_first`
* Released source archives now come GPG-signed with a detached `.asc` file

For the full list of closed issues see [the `iwyu 0.24` milestone][14].

Contributions in this release by Bolshakov, hexagon-recursion, JasonnnW3000, Kim Gräsman, kon72,
ruinansun. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.24.src.tar.gz][15]. It is equivalent
to the `0.24` tag and `clang_20` branch.

#### 10 Nov 2024

iwyu 0.23 compatible with llvm+clang 19 is released. Major changes:

* [iwyu] New policy for finding compiler built-in headers (breaking change)
* [iwyu] Treat types from overridden method signatures as provided by base
* [iwyu] Analyze associated headers more closely with their source file
* [iwyu] Many improvements for template analysis
* [iwyu] Accept -xc++-header
* [iwyu] Improve reporting of binary operators in macros
* [iwyu] Improve heuristics for reporting overloaded functions
* [iwyu] Consider variable definitions uses of extern declarations
* [mappings] Add mapping generator for GNU libstdc++
* [mappings] Regenerate mappings for GNU libstdc++ version 11
* [mappings] Improve dynamic `@headername` mappings
* [mappings] Update public standard library header list for C++23
* [mappings] Improve mappings for standard C library
* [iwyu_tool] Add new 'clang-warning' formatter
* [iwyu_tool] Default to system core count if -j is omitted

For the full list of closed issues see [the `iwyu 0.23` milestone][16].

Contributions in this release by Alejandro Colomar, Alfredo Correa, Bolshakov, Chris Down, firewave,
Jean-Philippe Gravel, Kim Gräsman, kon72, Michele Caini, Petar Vutov, Richard, scuzqy, ShalokShalom,
Thomas Tanner. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.23.src.tar.gz][17]. It is equivalent
to the `0.23` tag and `clang_19` branch.

#### 11 Mar 2024

iwyu 0.22 compatible with llvm+clang 18 is released. Major changes:

* [iwyu] Improve type analysis for typedefs, aliases and templates
* [iwyu] Improve analysis of macros expanding macros
* [iwyu] Improve IWYU driver for better validation and job handling
* [iwyu] Reject IWYU invocations with precompiled headers (see FAQ)
* [iwyu_tool] Better preserve failure exit codes
* [mappings] Add mappings for libstdc++ `<debug/...>` headers
* [mappings] Make mappings for POSIX and standard C headers stricter (more portable)
* [doc] Add separate FAQ page for longer-form practical user documentation
* [ci] IWYU now runs on itself in CI (finally!)

For the full list of closed issues see [the `iwyu 0.22` milestone][18].

Contributions in this release by Alejandro Colomar, Bolshakov, David Kilroy, Kim Gräsman, kon72.
Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.22.src.tar.gz][19]. It is equivalent
to the `0.22` tag and `clang_18` branch.

#### 08 Nov 2023

iwyu 0.21 compatible with llvm+clang 17 is released. Major changes:

* [iwyu] Improve analysis of type aliases (`typedef` and `using`)
* [iwyu] Improve analysis of namespace aliases (`namespace xyz = foobar`)
* [iwyu] Improve support for elaborated forward declarations (`typedef struct Foo Bar;`)
* [iwyu] Improve handling of "autocast" and function return types, particularly with complex
  template types.
* [iwyu] Add new `IWYU pragma: always_keep`, which lets a header announce that it should always be
  kept wherever included
* [iwyu] Automatically use builtin libc++ mappings if libc++ is the active standard library
* [mappings] Improve mappings for libc++ and posix headers

For the full list of closed issues see [the `iwyu 0.21` milestone][20].

Contributions in this release by Alejandro Colomar, Andrey Ali Khan Bolshakov, David Kilroy, Florian
Schmaus, Kim Gräsman. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.21.src.tar.gz][21]. It is equivalent
to the `0.21` tag and `clang_17` branch.

#### 02 Apr 2023

iwyu 0.20 compatible with llvm+clang 16 is released. Major changes:

* [iwyu] Support `IWYU pragma: export` for forward-declarations
* [iwyu] Silently break cycles in mappings instead of crashing
* [iwyu] Require full type inside `typeid()`
* [iwyu] Improve template reporting and resugaring
* [iwyu] Improve reporting of explicit template instantiations
* [iwyu] Fix a few crashers
* [iwyu] Improve logging (many small fixes)
* Abandon python2 for scripts in favor of python3

For the full list of closed issues see [the `iwyu 0.20` milestone][22].

Contributions in this release by Andrey Ali Khan Bolshakov, David Kilroy, Kim Gräsman, Matthew
Fennell, Petr Bred, Sameer Rahmani, Seth R. Johnson. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.20.src.tar.gz][23]. It is equivalent
to the `0.20` tag and `clang_16` branch.

#### 02 Nov 2022

iwyu 0.19 compatible with llvm+clang 15 is released. Major changes:

* [iwyu] New `--comment_style` option to control verbosity of 'why' comments
* [iwyu] New `--regex` option to select regex dialect
* [iwyu] Add support for regex replacement in mappings
* [iwyu] Add `begin_keep`/`end_keep` pragmas for protecting ranges of includes or forward-declares
* [iwyu] Fix several crasher bugs for unusual inputs
* [iwyu] More exhaustive handling of type aliases and enums
* [iwyu] Recognize IWYU pragmas in CRLF source files
* [iwyu] Respect configured toolchain on macOS (and overrides via `-nostdinc++` + `-isystem`)
* [fix_includes] Recognize namespace alias declarations
* [mappings] Improve mappings for POSIX and libc headers
* [cmake] Build now requires a C++17 compiler (as does LLVM)
* [cmake] Support LLVM external project build (see README)

For the full list of closed issues see [the `iwyu 0.19` milestone][24].

Contributions in this release by Aaron Puchert, Alejandro Colomar, Andrey Ali Khan Bolshakov, Boleyn
Su, Daniel Hannon, Et7f3, fanquake, Jan Kokemüller, Jean-Philippe Gravel, jspam, Kim Gräsman. Sorry
if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.19.src.tar.gz][25]. It is equivalent
to the `0.19` tag and `clang_15` branch.

#### 31 Mar 2022

iwyu 0.18 compatible with llvm+clang 14 is released. Major changes:

* [iwyu] Fix crash on C++20 consteval expressions
* [iwyu] Use more conventional exit codes (breaking change!)
* [iwyu_tool] Fix deprecation warning for python3
* [iwyu] Fix crash on va_list on AArch64
* [iwyu] Improved support for using-declarations based on new Clang design

Note the breaking change for exit codes; IWYU now always returns zero by default. See the README for
more information and manual overrides.

For the full list of closed issues see [the `iwyu 0.18` milestone][26].

Contributions in this release by Carlos Galvez and Kim Grasman. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.18.src.tar.gz][27]. It is equivalent
to the `0.18` tag and `clang_14` branch.

#### 05 Dec 2021

iwyu 0.17 compatible with llvm+clang 13 is released. Major changes:

* [iwyu] Improve support for various C++ features (builtins, CTAD, specializations, type aliases)
* [iwyu] Fix crash on invalid code
* [iwyu] Remove hard dependency on x86 LLVM target
* [mappings] Improve mappings for GNU libc
* [iwyu_tool] More concise output for clang output format

For the full list of closed issues see [the `iwyu 0.17` milestone][28].

Contributions in this release by Alejandro Colomar, Bolshakov, David Fetter, Kim Grasman, Omar
Sandoval, Salman Javed, Sven Panne. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.17.src.tar.gz][29]. It is equivalent
to the `0.17` tag and `clang_13` branch.

#### 26 May 2021

iwyu 0.16 compatible with llvm+clang 12 is released. Major changes:

* [iwyu_tool] Accept `--load/-l` argument for load limiting
* [iwyu_tool] Signal success/failure with exit code
* [mappings] Harmonize mapping generators
* [mappings] Add mapping generator for CPython
* [mappings] Improve mappings for libstdc++ and Boost
* [cmake] Add explicit C++14 compiler flag
* ... and many internal improvements

For the full list of closed issues see [the `iwyu 0.16` milestone][30].

Contributions in this release by Alexey Storozhev, Florian Schmaus, Kim Grasman, Omer Anson, saki7.
Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.16.src.tar.gz][31]. It is equivalent
to the `0.16` tag and `clang_12` branch.

#### 21 November 2020

iwyu 0.15 compatible with llvm+clang 11 is released. Major changes:

* [iwyu] Fix crash due to undefined behavior in AST traversal
* [iwyu] Improve handling of operator new including C++17 features
* [iwyu] Improve handling of templates
* [iwyu_tool] Remove known compiler wrappers from the command list
* [mappings] Improve Qt mapping generator
* [mappings] Improve boost mappings
* [mappings] Improve built-in mappings for `<time.h>`
* [mappings] Add built-in mappings for `max_align_t`, `ptrdiff_t`, and `wchar_t`
* [cmake] Support shared LLVM/Clang libraries and other improvements

For the full list of closed issues see [the `iwyu 0.15` milestone][32].

Contributions in this release by Andrea Bocci, David Callu, Emil Gedda, Florian Schmaus, John
Bytheway, Kim Grasman, Liam Keegan, Omar Sandoval, pppyx, Romain Geissler, Seth R Johnson, Tim
Gates. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.15.src.tar.gz][33]. It is equivalent
to the `0.15` tag and `clang_11` branch.

#### 17 May 2020

iwyu 0.14 compatible with llvm+clang 10 is released. Major changes:

* [iwyu] Report non-builtin enum base types
* [iwyu] Disable forward-declares for decls in inline namespaces
* [iwyu] Make C structs forward-declarable again
* [iwyu] Always keep Qt .moc includes
* [iwyu] Include binary type traits in analysis (e.g. `__is_convertible_to`)
* [iwyu_tool] Fail fast if include-what-you-use is not found
* [iwyu_tool] Print all diagnostic ouptut to stderr
* [fix_includes] Improve file extension detection
* Add man page for include-what-you-use

For the full list of closed issues see [the `iwyu 0.14` milestone][34].

Contributions in this release by Aaron Puchert, Kim Grasman, Miklos Vajna, Nick Overdijk, Uladzislau
Paulovich. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.14.src.tar.gz][35]. It is equivalent
to the `0.14` tag and `clang_10` branch.

#### 26 October 2019

iwyu 0.13 compatible with llvm+clang 9.0 is released. Major changes:

* [iwyu] Improved handling of relative includes in mappings and pragmas
* [iwyu] Path normalization now collapses `..`
* [iwyu] Improve `--no_fwd_decls` not to remove required forward declarations
* [iwyu] Improved handling of builtin templates
* [iwyu] Don't mark forward declarations `final`
* [iwyu] Tolerate `using` declarations in precompiled header
* [mappings] Add script to generate Qt mappings, and new mappings for Qt 5.11
* [iwyu_tool] Use `directory` from compilation database if available
* Numerous documentation and build improvements

For the full list of closed issues see [the `iwyu 0.13` milestone][36].

Contributions in this release by Alexander Grund, i-ky, John Bytheway, Julien Cabieces, Kim Grasman,
Levente Győző Lénárt, Miklos Vajna, Uladzislau Paulovich and Zachary Henkel. Sorry if we've missed
anyone.

The source code can be downloaded from [include-what-you-use-0.13.src.tar.gz][37]. It is equivalent
to the `0.13` tag and `clang_9.0` branch. Note that there's no `clang_9.0` tag this time, to avoid
tag/branch confusion.

#### 15 April 2019

iwyu 0.12 compatible with llvm+clang 8.0 is released. Major changes:

* [iwyu] New command-line option: `--keep` to mirror `IWYU pragma: keep`
* [iwyu] New command-line option: `--cxx17ns` to suggest compact C++17 nested namespaces
* [iwyu] Improve `--no_fwd_decls` to optimize for minimal number of redeclarations
* [iwyu] Improved mappings for POSIX types and let mappings apply to builtins as well
* [iwyu] More principled handling of explicit template instantiations
* [iwyu_tool] Breaking change: extra args are no longer automatically prefixed with `-Xiwyu` (so you
  can use them for Clang args too)
* [iwyu_tool] Better Windows support
* [fix_includes] Better handling of template forward-decls

For the full list of closed issues see
[https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.12%22][
38]

Contributions in this release by Asier Lacasta, David Robillard, Ignat Loskutov, Jakub Wilk, John
Bytheway, J.Ru, Kim Grasman, Martin Villagra, Miklos Vajna, tomKPZ, Tom Rix. Sorry if we've missed
anyone.

The source code can be downloaded from [include-what-you-use-0.12.src.tar.gz][39]. It is equivalent
to the `clang_8.0` tag.

#### 8 December 2018

iwyu 0.11 compatible with llvm+clang 7.0 is released. Major changes:

* [iwyu] Improved recognition of template and specialization uses
* [iwyu] Improved CMake build system, see docs for build instructions
* [mappings] Improved mappings for Boost, Intel intrinsics and libstdc++
* [iwyu_tool] Several bug fixes and improvements
* [iwyu_tool] Add `--basedir` argument to interpret IWYU output from another source tree
* [fix_includes] Handle namespaces better
* [fix_includes] Add `--only_re` switch to filter affected files
* [fix_includes] Add `--reorder/--noreorder` switch to toggle reordering of includes

For the full list of closed issues see
[https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.11%22][
40]

Contributions in this release by Asier Lacasta, Christian Venegas, Ignat Loskutov, J.Ru, Kim
Grasman, Martin Villagra, Paul Seyfert, Phantal, Philip Pfaffe, Scott Ramsby, Tom Rix, Victor
Poughon. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.11.src.tar.gz][41]. It is equivalent
to the `clang_7.0` tag.

#### 30 April 2018

iwyu 0.10 compatible with llvm+clang 6.0 is released. Major changes:

* Add `--no_fwd_decls` option to avoid replacing includes with forward-declarations
* Treat definitions of free functions as uses of the corresponding prototypes
* Support C++11 range-for loops
* Several template misattribution bugs fixed
* Better support for non-ASCII encodings in fix_includes.py
* Remove support for VCS commands from fix_includes.py

For the full list of closed issues see
[https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.10%22][
42]

Contributions in this release by bungeman, Kim Gräsman, Alex Kirchhoff, J. Ru, Orgad Shaneh,
Christoph Weiss. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.10.src.tar.gz][43]. It is equivalent
to the `clang_6.0` tag.

#### 11 March 2018

iwyu 0.9 compatible with llvm+clang 5.0 is released. Major changes:

* Improve handling of template arguments
* Improve support of JSON compilation databases (`arguments` field)
* Improve support for function pointers to templates
* Allow `IWYU pragma: keep` on forward declarations
* Fix a few crash scenarios on C++11 using-declarations
* iwyu_tool.py now supports parallel execution

For the full list of closed issues see
[https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.9%22+is
%3Aclosed][44]

Contributions in this release by J. Ru, Kim Gräsman, Kristoffer Henriksson, Paul Seyfert. Sorry if
we've missed anyone.

**NOTE:** From now on we will not be able to produce binary releases. There are well-maintained
packages for several platforms, and we rely on community contributions to increase availability
here.

The source code can be downloaded from [include-what-you-use-0.9.src.tar.gz][45]. It is equivalent
to the `clang_5.0` tag.

#### 3 August 2017

iwyu 0.8 compatible with llvm+clang 4.0.0 is released. Major changes:

* Add support for `IWYU pragma: associated`
* Better validation of pragma syntax in general.
* Improve support for out-of-tree builds, especially with MSVC.
* Add more compiler-like output for `iwyu_tool.py`
* Further improve location reporting in macros.
* Stricter requirements for arrays of templates.
* Better recognition of `typedef` types for by-value arguments.
* Better function pointers support.
* Documentation improvements.
* Extend `IWYU pragma: keep` to work with forward declarations.
* Fix Windows path handling in `fix_includes.py`
* Better libc++ container support.

For the full list of closed issues see
[https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.8%22+is
%3Aclosed][46]

Contributions in this release by Eugene Zelenko, ivankoster, Kim Gräsman, Kristoffer Henriksson,
mineo, nocnokneo, svenpanne, Volodymyr Sapsai, xuzhen1994. Sorry if we've missed anyone.

The source code can be downloaded from [include-what-you-use-0.8.src.tar.gz][47]. It is equivalent
to `clang_4.0` tag.

#### 30 October 2016

iwyu 0.7 compatible with llvm+clang 3.9 is released. Major changes:

* Add preliminary mappings for libc++.
* Require the complete type for pointer arithmetic.
* Recognize nested classes in friend declarations.
* Better handling of X-macros/textual includes.
* Better handling of self-checking private headers (that raise an `#error` if included directly).
* Improve IWYU's understanding of implicit include dirs; the current source file's dirname is always
  a candidate now.
* Add implicit include dirs for libc++ on Darwin targets.
* Lots of internal cleanup based on output from clang-tidy.
* Reduce logging strategically, to get more relevant output.

For the full list of closed issues see
[https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.7%22+is
%3Aclosed][48]

Thanks for all your contributions and help Bothari, Eugene Zelenko, Flamefire, Kim Gräsman. Sorry if
I've missed anyone.

The source code can be downloaded from [include-what-you-use-0.7.src.tar.gz][49]. It is equivalent
to `clang_3.9` tag.

#### 15 May 2016

iwyu 0.6 compatible with llvm+clang 3.8 is released. In this version we

* Added mappings for Qt 5.4.
* Added better analysis of uses in macros.
* Added `--no_comments` switch to suppress why-comments.
* Fixed bug with global namespace qualifier on friend declarations.
* Fixed bug in `fix_includes.py` generating invalid diff output.

For the full list of closed issues see
[https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.6%22+is
%3Aclosed][50]

Thanks for all your contributions and help JVApen, Kim Gräsman, Philip Pfaffe, pseyfert, realazthat,
Sylvestre Ledru, ThosRTanner. Sorry if I've missed anyone.

The source code can be downloaded from [include-what-you-use-0.6.src.tar.gz][51]. It is equivalent
to `clang_3.8` tag.

#### 17 December 2015

iwyu 0.5 compatible with llvm+clang 3.7 is released. In this version we

* Migrated to GitHub. It includes updated docs and improved testing infrastructure.
* Added Boost and Qt mappings.
* Have better support for using declarations.
* Allow `size_t` from multiple headers.
* Fixed handling includes with common path prefix.

For the full list of closed issues see
[https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.5%22+is
%3Aclosed][52]

Thanks for all your contributions Scott Howard, bungeman, tpltnt, Chris Glover, Kim Gräsman. And
thank you for all you help Jérémie Delaitre, Richard Thomson, dpunset, Earnestly, Dave Johansen,
ThosRTanner. Sorry if I've missed anyone.

The source code can be downloaded from [include-what-you-use-0.5.src.tar.gz][53]. It is equivalent
to `clang_3.7` tag.

#### 4 June 2015

iwyu 0.4 compatible with llvm+clang 3.6 is released. It contains the following changes:

* fix_includes.py compatible with Python 3.
* iwyu_tool.py to run include-what-you-use with compilation database.
* Various bugfixes.

For the full list of closed issues see
[https://code.google.com/p/include-what-you-use/issues/list?can=1&q=closed-after%3A2014%2F10%2F23+cl
osed-before%3A2015%2F6%2F2][54]

This release received many contributions and I want to thank SmSpillaz, Paul Redmond, Chris Glover,
Ryan Pavlik, showard314, Fabian Gruber, Kim Gräsman for your help. And thanks to Dave Johansen,
MMendez534, Sylvestre Ledru for packaging include-what-you-use. Sorry if I've missed anyone.

The source code can be downloaded from [include-what-you-use-0.4.src.tar.gz][55]. It is equivalent
to `clang_3.6` tag.

#### 25 November 2014

iwyu 0.3 compatible with llvm+clang 3.5 is released. In this version we have

* Added rudimentary support for C code.
* Improved MSVC support for templated code and precompiled headers.
* Added support for public STL #includes, which improves the IWYU experience for libc++ users.

For the full list of closed issues see
[https://code.google.com/p/include-what-you-use/issues/list?q=closed-after%3A2014%2F02%2F23&can=1][5
6]

The source code can be downloaded from [include-what-you-use-3.5.src.tar.gz][57]. It is equivalent
to `clang_3.5` tag.

#### 22 February 2014

iwyu version compatible with llvm+clang 3.4 is released. The source code can be downloaded from
[include-what-you-use-3.4.src.tar.gz][58]. It is equivalent to `clang_3.4` tag.

#### 11 August 2013

We are moving downloads to Google Drive. iwyu version compatible with llvm+clang 3.3 can be found at
[include-what-you-use-3.3.tar.gz][59]. It is equivalent to `clang_3.3` tag.

#### 6 December 2011

Now that clang 3.0 is out, I released a version of iwyu that works against clang 3.0. It is
equivalent to r330. It is available in the 'downloads' section on the side pane. To use, just `cd`
to `llvm/tools/clang/tools` in your llvm/clang tree, and untar `include-what-you-use-3.0-1.tar.gz`
from that location. Then cd to `include-what-you-use` and type `make`. (A cmakefile is also
available.) You can run `make check-iwyu` to run the iwyu test suite.

#### 24 June 2011

It was just pointed out to me the tarball I built against llvm+clang 2.9 doesn't actually compile
with llvm+clang 2.9. I must have made a mistake packaging it. I've tried again; according to my
tests, anyway, the new version works as it's supposed to.

#### 8 June 2011

I finally got around to releasing a tarball that builds against llvm+clang 2.9. See the 'downloads'
section on the side pane. This is a rather old version of iwyu at this point, so you'll do much
better to download a current clang+llvm and the svn-root version of include-what-you-use, and build
from that. See [README.txt][60] for more details.

#### 13 April 2011

Work has been continuing at a furious pace on include-what-you-use. It's definitely beta quality by
now. :-) Well, early beta. I've not been making regular releases, but the SVN repository is being
frequently updated, so don't take the lack of news here to mean a lack of activity.

#### 4 February 2011

I'm very pleased to announce the very-alpha, version 0.1 release of include-what-you-use. See the
wiki links on the right for instructions on how to download, install, and run include-what-you-use.

I'm releasing the code as it is now under a "release early and often" approach. It's still very
early in iwyu, and the program will probably have mistakes on any non-trivial piece of code.
Furthermore, it still has google-specific bits that may not make much sense in an opensource
release. This will all get fixed over time. Feel free to dig in and suggest patches to help the
fixing along!

If you want to follow the discussion on include-what-you-use, and/or keep up to date with changes,
subscribe to the [Google Group][61].

[1]: http://groups.google.com/group/include-what-you-use
[2]: https://github.com/include-what-you-use/include-what-you-use/issues
[3]: https://github.com/include-what-you-use/include-what-you-use
[4]: https://github.com/include-what-you-use/include-what-you-use/blob/master/CONTRIBUTING.md
[5]: https://github.com/include-what-you-use/include-what-you-use/blob/master/README.md
[6]: https://github.com/include-what-you-use/include-what-you-use/blob/master/docs/WhyIWYU.md
[7]: https://github.com/include-what-you-use/include-what-you-use/tree/master/docs
[8]: /downloads/
[9]: http://clang.llvm.org/get_started.html
[10]: https://github.com/include-what-you-use/include-what-you-use/blob/e8498e9/tests/bugs/README.md
[11]: https://github.com/include-what-you-use/include-what-you-use/blob/43901e1/docs/IWYUMappings.md
#command-line-switches-for-mapping-files
[12]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.25
%22
[13]: /downloads/include-what-you-use-0.25.src.tar.gz
[14]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.24
%22
[15]: /downloads/include-what-you-use-0.24.src.tar.gz
[16]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.23
%22
[17]: /downloads/include-what-you-use-0.23.src.tar.gz
[18]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.22
%22
[19]: /downloads/include-what-you-use-0.22.src.tar.gz
[20]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.21
%22
[21]: /downloads/include-what-you-use-0.21.src.tar.gz
[22]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.20
%22
[23]: /downloads/include-what-you-use-0.20.src.tar.gz
[24]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.19
%22
[25]: /downloads/include-what-you-use-0.19.src.tar.gz
[26]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.18
%22
[27]: /downloads/include-what-you-use-0.18.src.tar.gz
[28]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.17
%22
[29]: /downloads/include-what-you-use-0.17.src.tar.gz
[30]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.16
%22
[31]: /downloads/include-what-you-use-0.16.src.tar.gz
[32]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.15
%22
[33]: /downloads/include-what-you-use-0.15.src.tar.gz
[34]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.14
%22
[35]: /downloads/include-what-you-use-0.14.src.tar.gz
[36]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.13
%22
[37]: /downloads/include-what-you-use-0.13.src.tar.gz
[38]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.12
%22
[39]: /downloads/include-what-you-use-0.12.src.tar.gz
[40]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.11
%22
[41]: /downloads/include-what-you-use-0.11.src.tar.gz
[42]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.10
%22
[43]: /downloads/include-what-you-use-0.10.src.tar.gz
[44]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.9%
22+is%3Aclosed
[45]: /downloads/include-what-you-use-0.9.src.tar.gz
[46]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.8%
22+is%3Aclosed
[47]: /downloads/include-what-you-use-0.8.src.tar.gz
[48]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.7%
22+is%3Aclosed
[49]: /downloads/include-what-you-use-0.7.src.tar.gz
[50]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.6%
22+is%3Aclosed
[51]: /downloads/include-what-you-use-0.6.src.tar.gz
[52]: https://github.com/include-what-you-use/include-what-you-use/issues?q=milestone%3A%22iwyu+0.5%
22+is%3Aclosed
[53]: /downloads/include-what-you-use-0.5.src.tar.gz
[54]: https://code.google.com/p/include-what-you-use/issues/list?can=1&q=closed-after%3A2014%2F10%2F
23+closed-before%3A2015%2F6%2F2
[55]: http://include-what-you-use.org/downloads/include-what-you-use-0.4.src.tar.gz
[56]: https://code.google.com/p/include-what-you-use/issues/list?q=closed-after%3A2014%2F02%2F23&can
=1
[57]: http://include-what-you-use.com/downloads/include-what-you-use-3.5.src.tar.gz
[58]: http://include-what-you-use.com/downloads/include-what-you-use-3.4.src.tar.gz
[59]: https://docs.google.com/file/d/0ByBfuBCQcURXQktsT3ZjVmZtWkU/edit
[60]: http://code.google.com/p/include-what-you-use/source/browse/trunk/README.txt?r=260
[61]: http://groups.google.com/group/include-what-you-use
