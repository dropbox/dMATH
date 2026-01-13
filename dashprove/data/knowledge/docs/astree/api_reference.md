Astrée is a static analyzer for safety-critical software written or gen­er­ated in C or C++.

[[Astrée screenshot]][1]
[[Astrée screenshot]][2]
[[NIST logo]][3]

Astrée is sound — that is, if no errors are signaled, the absence of errors has been formally
proved. In 2020, the US National Institute of Standards and Technology determined that Astrée
[satisfies their criteria][4] for sound static code analysis.

### Features

* Exceptionally fast and precise analyses, including:
  
  * Runtime error analysis
  * Data race analysis
  * Non-interference analysis
  * Advanced taint analysis
  * Detection of Spectre vulnerabilities
  * Signal-flow, data-flow, and control-flow analysis
  * Control-coupling and component-interference analysis
  * Identification of functional and non-functional hazards according to norms such as ISO 26262,
    DO-178B/C, IEC-61508, and EN-50128
  * User-defined cybersecurity analyses for norms such as DO-356A and ISO 21434
  * [Rule checks][5] for MISRA, CERT, JSF, CWE, AUTOSAR, and other guidelines
  * Analysis of numerous [code metrics][6]
  * Dead code recognition
* Support for C, C++, and mixed code bases
* Support for collaborative reviews of analysis results
* TLS-encrypted communication with OpenID and OAuth2 support
* Delta analysis for easy evaluation of code revisions
* Automatically generated report files for documentation and certification purposes
* [Qualification Support Kits][7] for DO-178B/C, ISO 26262, EN-50128, and other standards
* [Feature-rich GUI][8] with graphical and textual views for analysis results, source code, control
  flow, and user comments
* Command-line mode for easy integration into automated build processes
* [Plugins][9] for TargetLink, μVision, and Jenkins; custom build for Raptor; LSP support for
  integration with any IDE or code editor that provides an LSP client, such as Visual Studio Code or
  Eclipse
* Node-locked, floating, and cloud licenses
* [Regular updates][10] and excellent tech support

### Who uses Astrée?

[Airbus logo]

Since 2003, Airbus France has been using Astrée in the devel­opment of safety-critical software for
vari­ous aircraft series, including the A380.

[Bosch logo]

In 2018, Bosch Automotive Steering [replaced their legacy tools][11] with Astrée and RuleChecker in
a pilot project for the entire Bosch Group. This led to significant savings thanks to faster
analyses, higher accuracy, and optimized licensing and support costs. Bosch then acquired a
worldwide license for both tools and the accompanying Qualification Support Kits.

[Framatome logo]

Framatome employs Astrée for verification of their safety-critical TELEPERM XS platform that is used
for engineering, testing, commissioning, operating and troubleshooting nuclear reactors.

[Helbako logo]

The global automotive supplier Helbako in Germany is using Astrée to guar­antee that no run­time
errors can occur in their elec­tronic control software and to de­monstrate [MISRA compliance][12] of
the code.

[ESA logo]

In 2008, Astrée proved the absence of any run­time errors in a C ver­sion of the auto­matic docking
software of the Jules Verne Automated Transfer Vehicle, enabling ESA to transport payloads to the
Inter­national Space Station.

[ebm-papst logo]

A world leader in motors and ventilators for air-conditioning and refrigeration systems, ebm-papst
is using Astrée for fully automatic conti­nuous verification of safety-critical interrupt-driven
control software for commu­tating high-efficiency EC motors for venti­lator systems.

[SYSGO logo]

In the EU research project [AQUAS][13], Astrée was used to analyze SYSGO’s PikeOS operating system.

[MTU Friedrichshafen logo]

MTU Friedrichshafen uses Astrée to demonstrate the cor­rectness of [con­trol software for emer­gency
power gene­rators][14] in power plants. Together with its quali­fication package, Astrée is part of
the IEC 60880 certi­fication process.

### Static analysis of runtime properties

[Astrée logo]

Astrée statically analyzes whether the programming language is used correctly and whether there can
be any run­time errors during any execution in any environment. This covers any use of C or C++ that,
according to the selected language standard, has undefined behavior or violates hardware-specific
aspects.

Additionally, Astrée reports invalid concurrent behavior, violations of user-specified programming
guidelines, and various program properties relevant for functional safety.

Astrée detects any:

* division by zero,
* out-of-bounds array indexing,
* erroneous pointer manipulation and dereferencing (NULL, uninitialized and dangling pointers),
* integer and floating-point arithmetic overflow,
* read access to uninitialized variables,
* data races (read/write or write/write concurrent accesses by two threads to the same memory
  location without proper mutex locking),
* inconsistent locking (lock/unlock problems),
* invalid calls to operating system services (e.g. OSEK calls to `TerminateTask` on a task with
  unreleased resources),
* Spectre vulnerabilities,
* other critical data and control-flow errors,
* violations of optional user-defined assertions to prove additional runtime properties (similar to
  assert diagnostics),
* dead code.

Astrée is sound for [floating-point computations][15] and handles them precisely and safely. All
possible rounding errors, and their cumu­lative effects, are taken into account. The same is true for
−∞, +∞ and `NaN` values and their effects through arithmetic calculations and comparisons.

### Data-flow and control-flow analysis

[Flow icon]

Astrée tracks accesses to global variables, static variables, and local variables whose accesses are
made outside of the frame in which the local variables are defined (e.g. because their address is
passed into a called function).

All data and function pointers are resolved automatically. The soundness of the analysis ensures
that all potential targets of data and function pointers are taken into account.

### Visualization of control-flow, data-flow & signal-flow graphs

A picture is worth a thousand words. The graph-visualization capabilities of Astrée help you
discover unexpected behavior of your software, and enable a faster and deeper understanding of
third-party or legacy code.

### Signal-flow analysis

Astrée is able to analyze the flow from input signals to output signals through complex real-world
software.

This is accomplished through taint analysis that builds upon a full-fledged semantic analysis, thus
taking into account any changes to the signal flow by the configuration parameters of your
application.

The analysis can be performed for a specific application configuration, avoiding false signal flows
not present in that configuration. Alternatively, the influence of application parameters on the
output signal can be determined for a generic configuration.

If the taint analysis does not report an influence of an input signal on an output signal,
independence has been proven. If an unexpected influence is reported, Astrée helps you understand
the signal flow to either improve your code or tweak the analysis precision.

### Analysis of interference between software components

[Icon]

Astrée enables you to define software components according to your own criteria, and to specify
which data or control flow between the components is of interest to you.

Astrée then detects every potential data and control interference between the components at the
source-code level, and helps you understand the causes to improve your code as needed.

### Tailor it to your own requirements

Further kinds of external knowledge can be supplied to the analyzer, e.g. to fine-tune its precision
for individual loops or data structures.

This allows for analyses with very few or even zero false alarms.

Detailed messages and an intui­tive GUI guide you to the exact cause of each potential runtime error.
Actual errors can be fixed straight away, and false alarms can be tuned out for all subsequent
analyses.

### Take it to the next level

[[CompCert logo]][16]

Once your C code is error-free, you can use our formally verified optimizing C compiler
[CompCert][17] to guarantee that all the safety properties verified on your source code also hold
for your machine code. CompCert is the only production compiler that is mathematically proven to be
free of mis­compilation issues.

### Support for C++ and mixed code bases

[C++ icon]

Since 2020, Astrée can be applied to C++ and mixed C/C++ code bases. By now, it supports C++
versions from C++98 to C++23.

Astrée’s C++ analysis is designed to meet the characteristics of safety-critical embedded software
and so is subject to the same restrictions as Astrée for C.

The high-level abstraction features and template library of C++ facilitate the design of very
complex and dynamic software. Extensive use of these features may violate the established principles
of safety-critical embedded software development and lead to unsatis­fac­tory analysis times and
results. The Astrée manual gives recommendations on the use of C++ features to ensure that the code
can be well analyzed. For less constrained (less critical) C++ code, we recommend using the
standalone RuleChecker.

### MISRA and more

[[RuleChecker logo]][18]

The seamlessly integrated [RuleChecker][19] lets you check your code for compliance with various
coding standards, including [MISRA][20], [CWE][21], [ISO/IEC][22], [SEI CERT][23], and
[AUTOSAR][24]. You can easily toggle individual rules and even specific aspects of certain rules.
The tool can also check for various code metrics such as com­ment den­sity or cyclomatic complexity.
Custom extensions for your own in-house coding guidelines are available on request.

RuleChecker can be envoked separately, to allow for even faster checks of your code, or in
conjunction with the sound semantic analyses offered by Astrée, to additionally guarantee zero false
negatives and minimize false positives on semantical rules. No competing standalone MISRA checker
can offer this, and no testing environment can match the full data and path coverage provided by the
static analysis.

### Delta analysis

With Astrée, you can quickly and fully understand the impact of any changes that you make to your
code. Various charts and tables clearly explain any differences in performance between code
revisions.

### Easy integration into your workflow

[[dSPACE logo]][25] [[Jenkins logo]][26] [[Eclipse logo]][27]

The analyzer can also run in batch mode for easy integration into established tool-chains.

Plugins for [TargetLink][28] and [Jenkins][29] are available and actively maintained. LSP is
supported for easy integration into Visual Studio Code, [Eclipse][30] or any other IDE or code
editor that provides an LSP client.

### Qualification support

[QSK icon]

Your usage of Astrée can be qualified according to DO-178B/C, ISO 26262, IEC 61508, EN-50128, the
FDA Principles of Software Validation, and other safety standards. We offer special [Qualification
Support Kits][31] that simplify and automate the qualification process.

### Ten years ahead of the competition

Astrée is…

* Sound
* Automatic
* Fast
* Domain-aware
* Parametric
* Modular
* Precise
* Up-to-date

* #### Sound
  
  Most static analyzers do not consider all possible run­time errors. Others specifically focus on
  the most probable ones.
  
  As a result, almost all competing tools can only be used for testing (i.e. finding certain kinds
  of bugs), but never for verification (i.e. proving the absence of any run­time errors).
  
  In contrast, Astrée is *sound*. It always exhaustively considers all pos­sible run­time errors. At
  the same time, it is capable of producing exactly [zero false alarms][32]. This is crucial for
  verification of safety-critical software.
  
  In 2020, the US National Institute of Standards and Technology [determined][33] Astrée to be one
  out of only two tools in total that satisfy their criteria for sound static code analysis.
* #### Automatic
  
  Certain types of static analyzers, e.g. those relying on theorem provers, require programs to be
  annotated with lots of inductive invariants.
  
  Astrée usually requires very few annotations. On some programs, it can even run *completely
  automatically*, without any help from the user.
  
  Many analyzers cannot be scripted at all. Others can, but won’t let you access their analysis
  results outside of their proprietary viewer. This actively prevents you from automating the
  analysis, e.g. as part of your nightly build process.
  
  In contrast, Astrée offers you complete access to the analysis engine in batch mode, and lets you
  freely export the analysis results and further process them in any way you choose, no strings
  attached.
* #### Fast
  
  Many static analyzers have high computational costs (typically, sever­al hours of compu­tation per
  10,000 lines of code); others terminate out of memory, or may not terminate at all.
  
  In contrast, Astrée is *efficient* and easily scales up to real-world pro­grams in industrial
  practice.
  
  As an example, in order to ana­lyze actual flight-control software with 132,000 lines of C code,
  even on a slow 2.8GHz PC Astrée takes a mere 80 min­utes. Faster machines will get you faster
  results. Multi­core parallel or distri­buted computation is supported.
* #### Domain-aware
  
  General-purpose static analyzers aim at analyzing any application written in a given pro­gram­ming
  language. They can rely on language related properties to find potential run­time errors.
  Specialized static analyzers put additional restrictions on the app­lications so as to be able to
  take specific program structures into account.
  
  In contrast, Astrée is *domain-aware*. It thus knows facts about ap­pli­ca­tion domains that are
  indispensable to make sophisticated proofs. For example, for control/command programs, Astrée
  takes the logic and functional properties of control/command theory into account.
* #### Parametric
  
  In static program analysis, there is always a trade-off between analysis precision and analysis
  cost. Analyzers that are precise are usually also very slow, while fast analyzers usually lack
  precision.
  
  In contrast, Astrée is *parametric*, allowing you to freely trade speed for precision and vice
  versa. The level of abstraction used for analysis can be easily tailored to your very own
  requirements.
* #### Modular
  
  Astrée consists of several modules — so-called *abstract domains*. These modules can be assembled
  and parameterized to build application-specific analyzers that are fully adapted to a particular
  ap­plication domain or specific end-user requirements.
  
  In case of false alarms, Astrée can be easily extended by intro­duc­ing additional modules that
  enhance the precision of the analysis.
* #### Precise
  
  General-purpose static analyzers usually suffer from low precision in terms of false alarms, i.e.
  spurious warnings about errors that can actu­ally never occur at run­time. The ratio of false alarms
  to the number of basic C operations typically ranges between 10% and 20%.
  
  Specialized analyzers achieve a better precision of 10% or less. How­ever, even a very high
  selectivity rate of only 1 false alarm for every 100 operations is usually unacceptable for large
  real-world appli­cations. For example, on a program with 100,000 operations, a selec­tivity rate of
  only 1% yields 1000 false alarms.
  
  In contrast, thanks to its modularity and domain-awareness, Astrée can be made *exceptionally
  precise*, often to the point of pro­ducing exactly *zero false alarms*. This has been repeatedly
  proven in indus­tri­al practice, e.g. when analyzing primary flight-control software for Airbus.
* #### Up-to-date
  
  Astrée builds upon decades of research in static program analysis, and also makes sure to
  incorporate the latest ongoing research, always staying well ahead of any competition.
  
  Major new releases are published [twice a year][34], intermediate releases more often still. If
  you have a feature request, let us know at [support@absint.com][35].


### Free trial

[Start your free trial today][36], complete with online training and technical support.

#### Further information

* [Workflow, code examples, user interface][37]
* [Compliance matrices for MISRA, CWE, AUTOSAR…][38]
* [Integration with TargetLink, Jenkins, Eclipse…][39]
* [Current release: 25.10][40]

#### PDF downloads

* [Product flyer][41]
* [Datasheet][42]

#### Get in touch

* [info@absint.com][43]
* [Our distributors][44]
* [Upcoming events][45]

[1]: gallery.htm#shot6
[2]: gallery.htm#shot4
[3]: compliance.htm#nist
[4]: compliance.htm#nist
[5]: ../rulechecker/
[6]: compliance.htm#metrics
[7]: ../qualification/
[8]: gallery.htm
[9]: targetlink.htm
[10]: ../releasenotes/astree/
[11]: ../bosch_as.htm
[12]: compliance.htm#misra
[13]: ../projects.htm#completed
[14]: ../mtu_fh.htm
[15]: examples.htm#floatingpoint
[16]: ../compcert/
[17]: ../compcert/
[18]: ../rulechecker/
[19]: ../rulechecker/
[20]: compliance.htm#misra
[21]: compliance.htm#cwe
[22]: compliance.htm#iso
[23]: compliance.htm#cert
[24]: compliance.htm#autosar
[25]: targetlink.htm
[26]: jenkins.htm
[27]: eclipse.htm
[28]: targetlink.htm
[29]: jenkins.htm
[30]: eclipse.htm
[31]: ../qualification/
[32]: javascript:vTab(1,7,8)
[33]: compliance.htm#nist
[34]: ../releasenotes/astree/
[35]: mailto:support@absint.com
[36]: contact.htm
[37]: workflow.htm
[38]: compliance.htm
[39]: targetlink.htm
[40]: ../releasenotes/astree/25.10/
[41]: ../flyers/Astree.pdf
[42]: ../factsheets/factsheet_astree_c_web.pdf
[43]: mailto:info@absint.com
[44]: ../shop/wheretobuy.htm
[45]: ../events.htm
