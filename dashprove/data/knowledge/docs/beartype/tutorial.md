Tip

💗 **Upbear us** at [GitHub Sponsors][1] and [SonarQube Advanced Security (Tidelift)][2]. **Follow
us** [on Bluesky][3]. **Friendzone us** [at Zulip][4]. Your generous support is our quality
assurance. 💗

[[beartype —[ the bare-metal type-checker ]—]][5]

[[beartype test coverage status]][6] [[beartype continuous integration (CI) status]][7] [[beartype
Read The Docs (RTD) status]][8]

**Beartype** is an [open-source][9] [pure-Python][10] [PEP-compliant][11] [near-real-time][12]
[hybrid runtime-static][13] [third-generation][14] [type-checker][15] emphasizing efficiency,
usability, unsubstantiated jargon we just made up, and thrilling puns.

Beartype enforces [type hints][16] across your entire app in [two lines of runtime code with no
runtime overhead][17]. If seeing is believing, prepare to do both those things.

# Install beartype.
$ pip3 install beartype

# Edit the "{your_package}.__init__" submodule with your favourite IDE.
$ vim {your_package}/__init__.py      # <-- so, i see that you too vim
# At the very top of your "{your_package}.__init__" submodule:
from beartype.claw import beartype_this_package  # <-- boilerplate for victory
beartype_this_package()                          # <-- yay! your team just won

Beartype now implicitly type-checks *all* annotated classes, callables, and variable assignments
across *all* submodules of your package. Congrats. This day all bugs die.

But why stop at the burning tires in only *your* code? Your app depends on a sprawling ghetto of
other packages, modules, and services. How riddled with infectious diseases is *that* code? You’re
about to find out.

# ....................{ BIG BEAR                        }....................
# Warn about type hint violations in *OTHER* packages outside your control;
# only raise exceptions from violations in your package under your control.
# Again, at the very top of your "{your_package}.__init__" submodule:
from beartype import BeartypeConf                              # <-- this isn't your fault
from beartype.claw import beartype_all, beartype_this_package  # <-- you didn't sign up for this
beartype_this_package()                                        # <-- raise exceptions in your code
beartype_all(conf=BeartypeConf(violation_type=UserWarning))    # <-- emit warnings from other code

Beartype now implicitly type-checks *all* annotated classes, callables, and variable assignments
across *all* submodules of *all* packages. When **your** package violates type safety, beartype
raises an exception. When any **other** package violates type safety, beartype just emits a warning.
The triumphal fanfare you hear is probably your userbase cheering. This is how the QA was won.

Beartype also publishes a [plethora of APIs for fine-grained control over type-checking][18]. For
those who are about to QA, beartype salutes you. Would you like to know more?

# So let's do this.
$ python3
# ....................{ RAISE THE PAW                   }....................
# Manually enforce type hints across individual classes and callables.
# Do this only if you want a(nother) repetitive stress injury.

# Import the @beartype decorator.
>>> from beartype import beartype      # <-- eponymous import; it's eponymous

# Annotate @beartype-decorated classes and callables with type hints.
>>> @beartype                          # <-- you too will believe in magic
... def quote_wiggum(lines: list[str]) -> None:
...     print('“{}”\n\t— Police Chief Wiggum'.format("\n ".join(lines)))

# Call those callables with valid parameters.
>>> quote_wiggum(["Okay, folks. Show's over!", " Nothing to see here. Show's…",])
“Okay, folks. Show's over!
 Nothing to see here. Show's…”
   — Police Chief Wiggum

# Call those callables with invalid parameters.
>>> quote_wiggum([b"Oh, my God! A horrible plane crash!", b"Hey, everybody! Get a load of this flami
ng wreckage!",])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 30, in quote_wiggum
  File "/home/springfield/beartype/lib/python3.9/site-packages/beartype/_decor/_code/_pep/_error/err
ormain.py", line 220, in get_beartype_violation
    raise exception_cls(
beartype.roar.BeartypeCallHintParamViolation: @beartyped
quote_wiggum() parameter lines=[b'Oh, my God! A horrible plane
crash!', b'Hey, everybody! Get a load of thi...'] violates type hint
list[str], as list item 0 value b'Oh, my God! A horrible plane crash!'
not str.

# ....................{ MAKE IT SO                      }....................
# Squash bugs by refining type hints with @beartype validators.
>>> from beartype.vale import Is  # <---- validator factory
>>> from typing import Annotated  # <---------------- if Python ≥ 3.9.0
# >>> from typing_extensions import Annotated   # <-- if Python < 3.9.0

# Validators are type hints constrained by lambda functions.
>>> ListOfStrings = Annotated[  # <----- type hint matching non-empty list of strings
...     list[str],  # <----------------- type hint matching possibly empty list of strings
...     Is[lambda lst: bool(lst)]  # <-- lambda matching non-empty object
... ]

# Annotate @beartype-decorated callables with validators.
>>> @beartype
... def quote_wiggum_safer(lines: ListOfStrings) -> None:
...     print('“{}”\n\t— Police Chief Wiggum'.format("\n ".join(lines)))

# Call those callables with invalid parameters.
>>> quote_wiggum_safer([])
beartype.roar.BeartypeCallHintParamViolation: @beartyped
quote_wiggum_safer() parameter lines=[] violates type hint
typing.Annotated[list[str], Is[lambda lst: bool(lst)]], as value []
violates validator Is[lambda lst: bool(lst)].

# ....................{ AT ANY TIME                     }....................
# Type-check anything against any type hint – anywhere at anytime.
>>> from beartype.door import (
...     is_bearable,  # <-------- like "isinstance(...)"
...     die_if_unbearable,  # <-- like "assert isinstance(...)"
... )
>>> is_bearable(['The', 'goggles', 'do', 'nothing.'], list[str])
True
>>> die_if_unbearable([0xCAFEBEEF, 0x8BADF00D], ListOfStrings)
beartype.roar.BeartypeDoorHintViolation: Object [3405692655, 2343432205]
violates type hint typing.Annotated[list[str], Is[lambda lst: bool(lst)]],
as list index 0 item 3405692655 not instance of str.

# ....................{ GO TO PLAID                     }....................
# Type-check anything in around 1µs (one millionth of a second) – including
# this list of one million 2-tuples of NumPy arrays.
>>> from beartype.door import is_bearable
>>> from numpy import array, ndarray
>>> data = [(array(i), array(i)) for i in range(1000000)]
>>> %time is_bearable(data, list[tuple[ndarray, ndarray]])
    CPU times: user 31 µs, sys: 2 µs, total: 33 µs
    Wall time: 36.7 µs
True

# ....................{ MAKE US DO IT                   }....................
# Don't know type hints? Do but wish you didn't? What if somebody else could
# write your type hints for you? @beartype: it's somebody. Let BeartypeAI™
# write your type hints for you. When you no longer care, call BeartypeAI™.
>>> from beartype.bite import infer_hint  # <----- caring begins

# What type hint describes the root state of a Pygments lexer, BeartypeAI™?
>>> from pygments.lexers import PythonLexer
>>> infer_hint(PythonLexer().tokens["root"])
list[
    tuple[str | pygments.token._TokenType[str], ...] |
    tuple[str | collections.abc.Callable[
        typing.Concatenate[object, object, ...], object], ...] |
    typing.Annotated[
        collections.abc.Collection[str],
        beartype.vale.IsInstance[pygments.lexer.include]]
]  # <---- caring ends

# ...all righty then. Guess I'll just take your word for that, BeartypeAI™.

Beartype brings [Rust][19]- and [C++][20]-inspired [zero-cost abstractions][21] into the lawless
world of [dynamically-typed][22] Python by [enforcing type safety at the granular level of functions
and methods][23] against [type hints standardized by the Python community][24] in \(O(1)\)
[non-amortized worst-case time with negligible constant factors][25]. If the prior sentence was
unreadable jargon, see [our friendly and approachable FAQ for a human-readable synopsis][26].

Beartype is [portably implemented][27] in [Python 3][28], [continuously stress-tested][29] via
[GitHub Actions][30] **×** [tox][31] **×** [pytest][32] **×** [Codecov][33], and [permissively
distributed][34] under the [MIT license][35]. Beartype has *no* runtime dependencies, [only one
test-time dependency][36], and [only one documentation-time dependency][37]. Beartype supports [all
actively developed Python versions][38], [all Python package managers][39], and [multiple
platform-specific package managers][40].

Beartype [powers quality assurance across the Python ecosystem][41].

# The Typing Tree[¶][42]

Welcome to the **Bearpedia** – your one-stop Encyclopedia Beartanica for all things @beartype. It’s
“[typing][43] or bust!” as you…

Bear with Us

* [Bearpedia][44]
* [Install][45]
  
  * [Platform][46]
    
    * [macOS][47]
    * [Arch Linux][48]
    * [Gentoo Linux][49]
  * [Badge][50]
* [tl;dr][51]
* [ELI5][52]
  
  * [Comparison][53]
    
    * […versus Static Type-checkers][54]
    * […versus Runtime Type-checkers][55]
  * [Quickstart][56]
    
    * [Standard Hints][57]
      
      * [Toy Example][58]
      * [Industrial Example][59]
  * [Tutorial][60]
    
    * [Builtin Types][61]
    * [Arbitrary Types][62]
    * [Unions of Types][63]
    * [Optional Types][64]
  * [Would You Like to Know More?][65]
* [API][66]
  
  * [The Left-Paw Path][67]
* [FAQ][68]
  
  * [What is beartype?][69]
  * [What is typeguard?][70]
  * [When should I use beartype?][71]
  * [Does beartype do any bad stuff?][72]
  * [Does beartype actually do anything?][73]
  * [How much does all this *really* cost?][74]
  * [Beartype just does random stuff? Really?][75]
  * [What does “pure-Python” mean?][76]
  * [What does “near-real-time” even mean? Are you just making stuff up?][77]
  * [What does “hybrid runtime-static” mean? Pretty sure you made that up, too.][78]
  * [“Third-generation type-checker” doesn’t mean anything, does it?][79]
  * [How do I type-check…][80]
    
    * […Boto3 types?][81]
    * […JAX arrays?][82]
    * […NumPy arrays?][83]
    * […PyTorch tensors?][84]
    * […mock types?][85]
    * […pandas data frames?][86]
    * […the current class?][87]
    * […under VSCode?][88]
    * […under [insert-IDE-name-here]?][89]
    * […with type narrowing?][90]
  * [How do I *ONLY* type-check while running my test suite?][91]
  * [How do I *NOT* type-check something?][92]
  * [Why is @leycec’s poorly insulated cottage in the Canadian wilderness so cold?][93]
* [Features][94]
* [Code][95]
  
  * [Beartype Code Generation: It’s All for You][96]
    
    * [Identity Decoration][97]
    * [Unconditional Identity Decoration][98]
    * [Shallow Identity Decoration][99]
    * [Deep Identity Decoration][100]
    * [Constant Decoration][101]
      
      * [Constant Builtin Type Decoration][102]
      * [Constant Non-Builtin Type Decoration][103]
      * [Constant Shallow Sequence Decoration][104]
      * [Constant Deep Sequence Decoration][105]
      * [Constant Nested Deep Sequence Decoration][106]
* [Beartype Dev Handbook: It’s Handy][107]
  
  * [Dev Workflow][108]
  * [Moar Depth][109]
  * [Moar Compliance][110]
* [Math][111]
  
  * [Beartype Timings][112]
    
    * [Timings Overview][113]
    * [Timings Lower Bound][114]
      
      * [Formulaic Formulas: They’re Back in Fashion][115]
      * [Function Call Overhead: The New Glass Ceiling][116]
      * [Holy Balls of Flaming Dumpster Fires][117]
      * [But, But… That’s Not Good Enough!][118]
  * [Nobody Expects the Linearithmic Time][119]
* [Moar][120]
  
  * [Runtime Type Checkers][121]
  * [Runtime Data Validators][122]
  * [Static Type Checkers][123]

*Let’s type this.*

# See Also[¶][124]

Beartype plugins adjacent to your interests include:

* [ipython-beartype][125], beartype’s official [IPython][126] plugin. Type-check:
  
  * Browser-based [Jupyter][127], [Marimo][128], and [Google Colab][129] notebook cells.
  * IDE-based [Zasper][130] notebook cells.
  * Terminal-based [IPython][131] REPLs.
* [pytest-beartype][132], beartype’s official [pytest][133] plugin. Type-check packages *only* at
  [pytest][134] test-time. Fatally obsessed with speed? Fatally accepting of critical failure? Can’t
  bear to type-check at runtime? When your team lacks trust, your team chooses
  [pytest-beartype][135].

# License[¶][136]

Beartype is [open-source software released][137] under the [permissive MIT license][138].

# Security[¶][139]

Beartype encourages security researchers, institutes, and concerned netizens to [responsibly
disclose security vulnerabilities as GitHub-originated Security Advisories][140] – published with
full acknowledgement in the public [GitHub Advisory Database][141].

# Funding[¶][142]

Beartype is financed as a [purely volunteer open-source project via GitHub Sponsors][143], to whom
our burgeoning community is eternally indebted. Without your generosity, runtime type-checking would
be a shadow of its current hulking bulk. We genuflect before your selfless charity, everyone!

Prior official funding sources (*yes, they once existed*) include:

1. A [Paul Allen Discovery Center award][144] from the [Paul G. Allen Frontiers Group][145] under
   the administrative purview of the [Paul Allen Discovery Center][146] at [Tufts University][147]
   over the period 2015—2018 preceding the untimely death of [Microsoft co-founder Paul Allen][148],
   during which beartype was maintained as the private `@type_check` decorator in the [Bioelectric
   Tissue Simulation Engine (BETSE)][149]. ^{Phew!}

# Contributors[¶][150]

Beartype is the work product of volunteer enthusiasm, excess caffeine, and sleepless Wednesday
evenings. These brave GitHubbers hurtled [the pull request (PR) gauntlet][151] so that you wouldn’t
have to:

[[Beartype contributors]][152]

It’s a heavy weight they bear. Applaud them as they buckle under the load!

# History[¶][153]

Beartype’s histrionic past is checkered with drama, papered over in propaganda, and chock full of
the stuff of stars. Gaze upon their glistening visage as they grow monotonically. But do the stars
matter? Neither to mortal nor to bear. Yet, by starlight, we all howl to commit by dawn.

[[Beartype stargazers]][154]

[

next

Install

** ][155]

[1]: https://github.com/sponsors/leycec
[2]: https://www.sonarsource.com/products/sonarqube/advanced-security
[3]: https://leycec.bsky.social
[4]: https://beartype.zulipchat.com
[5]: https://github.com/beartype/beartype
[6]: https://codecov.io/gh/beartype/beartype
[7]: https://github.com/beartype/beartype/actions?workflow=tests
[8]: https://beartype.readthedocs.io/en/latest/?badge=latest
[9]: https://github.com/beartype/beartype/blob/main/LICENSE
[10]: faq/#faq-pure
[11]: pep/#pep-pep
[12]: faq/#faq-realtime
[13]: faq/#faq-hybrid
[14]: faq/#faq-third
[15]: eli5/#eli5-eli5
[16]: eli5/#eli5-typing
[17]: api_claw/#api-claw-api-claw
[18]: api/#api-api
[19]: https://www.rust-lang.org
[20]: https://en.wikipedia.org/wiki/C%2B%2B
[21]: https://boats.gitlab.io/blog/post/zero-cost-abstractions
[22]: https://en.wikipedia.org/wiki/Type_system
[23]: eli5/#eli5-eli5
[24]: pep/#pep-pep
[25]: math/#math-time
[26]: faq/#faq-faq
[27]: https://github.com/beartype/beartype/tree/main/beartype
[28]: https://www.python.org
[29]: https://github.com/beartype/beartype/actions?workflow=tests
[30]: https://github.com/features/actions
[31]: https://tox.readthedocs.io
[32]: https://docs.pytest.org
[33]: https://about.codecov.io
[34]: https://github.com/beartype/beartype/blob/main/LICENSE
[35]: https://opensource.org/licenses/MIT
[36]: https://docs.pytest.org
[37]: https://www.sphinx-doc.org
[38]: https://devguide.python.org/versions/#versions
[39]: install/#install
[40]: install/#install
[41]: https://github.com/beartype/beartype/network/dependents
[42]: #the-typing-tree
[43]: https://docs.python.org/3/library/typing.html
[44]: #
[45]: install/
[46]: install/#platform
[47]: install/#macos
[48]: install/#arch-linux
[49]: install/#gentoo-linux
[50]: install/#badge
[51]: tldr/
[52]: eli5/
[53]: eli5/#comparison
[54]: eli5/#versus-static-type-checkers
[55]: eli5/#versus-runtime-type-checkers
[56]: eli5/#quickstart
[57]: eli5/#standard-hints
[58]: eli5/#toy-example
[59]: eli5/#industrial-example
[60]: eli5/#tutorial
[61]: eli5/#builtin-types
[62]: eli5/#arbitrary-types
[63]: eli5/#unions-of-types
[64]: eli5/#optional-types
[65]: eli5/#would-you-like-to-know-more
[66]: api/
[67]: api/#the-left-paw-path
[68]: faq/
[69]: faq/#what-is-beartype
[70]: faq/#what-is-typeguard
[71]: faq/#when-should-i-use-beartype
[72]: faq/#does-beartype-do-any-bad-stuff
[73]: faq/#does-beartype-actually-do-anything
[74]: faq/#how-much-does-all-this-really-cost
[75]: faq/#beartype-just-does-random-stuff-really
[76]: faq/#what-does-pure-python-mean
[77]: faq/#what-does-near-real-time-even-mean-are-you-just-making-stuff-up
[78]: faq/#what-does-hybrid-runtime-static-mean-pretty-sure-you-made-that-up-too
[79]: faq/#third-generation-type-checker-doesn-t-mean-anything-does-it
[80]: faq/#how-do-i-type-check
[81]: faq/#boto3-types
[82]: faq/#jax-arrays
[83]: faq/#numpy-arrays
[84]: faq/#pytorch-tensors
[85]: faq/#mock-types
[86]: faq/#pandas-data-frames
[87]: faq/#the-current-class
[88]: faq/#under-vscode
[89]: faq/#under-insert-ide-name-here
[90]: faq/#with-type-narrowing
[91]: faq/#how-do-i-only-type-check-while-running-my-test-suite
[92]: faq/#how-do-i-not-type-check-something
[93]: faq/#why-is-leycec-s-poorly-insulated-cottage-in-the-canadian-wilderness-so-cold
[94]: pep/
[95]: code/
[96]: code/#beartype-code-generation-it-s-all-for-you
[97]: code/#identity-decoration
[98]: code/#unconditional-identity-decoration
[99]: code/#shallow-identity-decoration
[100]: code/#deep-identity-decoration
[101]: code/#constant-decoration
[102]: code/#constant-builtin-type-decoration
[103]: code/#constant-non-builtin-type-decoration
[104]: code/#constant-shallow-sequence-decoration
[105]: code/#constant-deep-sequence-decoration
[106]: code/#constant-nested-deep-sequence-decoration
[107]: code/#beartype-dev-handbook-it-s-handy
[108]: code/#dev-workflow
[109]: code/#moar-depth
[110]: code/#moar-compliance
[111]: math/
[112]: math/#beartype-timings
[113]: math/#timings-overview
[114]: math/#timings-lower-bound
[115]: math/#formulaic-formulas-they-re-back-in-fashion
[116]: math/#function-call-overhead-the-new-glass-ceiling
[117]: math/#holy-balls-of-flaming-dumpster-fires
[118]: math/#but-but-that-s-not-good-enough
[119]: math/#nobody-expects-the-linearithmic-time
[120]: moar/
[121]: moar/#runtime-type-checkers
[122]: moar/#runtime-data-validators
[123]: moar/#static-type-checkers
[124]: #see-also
[125]: https://pypi.org/project/ipython-beartype
[126]: https://ipython.org
[127]: https://jupyter.org
[128]: https://marimo.io
[129]: https://colab.research.google.com
[130]: https://zasper.io
[131]: https://ipython.org
[132]: https://pypi.org/project/pytest-beartype
[133]: https://docs.pytest.org
[134]: https://docs.pytest.org
[135]: https://pypi.org/project/pytest-beartype
[136]: #license
[137]: https://github.com/beartype/beartype/blob/main/LICENSE
[138]: https://opensource.org/licenses/MIT
[139]: #security
[140]: https://github.com/beartype/beartype/blob/main/.github/SECURITY.md
[141]: https://github.com/advisories
[142]: #funding
[143]: https://github.com/sponsors/leycec
[144]: https://www.alleninstitute.org/what-we-do/frontiers-group/news-press/press-resources/press-re
leases/paul-g-allen-frontiers-group-announces-allen-discovery-center-tufts-university
[145]: https://www.alleninstitute.org/what-we-do/frontiers-group
[146]: http://www.alleninstitute.org/what-we-do/frontiers-group/discovery-centers/allen-discovery-ce
nter-tufts-university
[147]: https://www.tufts.edu
[148]: https://en.wikipedia.org/wiki/Paul_Allen
[149]: https://github.com/betsee/betse
[150]: #contributors
[151]: https://github.com/beartype/beartype/pulls
[152]: https://github.com/beartype/beartype/graphs/contributors
[153]: #history
[154]: https://github.com/beartype/beartype/stargazers
[155]: install/
