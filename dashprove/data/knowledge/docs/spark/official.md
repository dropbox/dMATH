[Spark code snippet 2]
Languages > SPARK_

# The SPARK Programming Language

SPARK, which is based on Ada, offers unparalleled safety and security through its design and support
for deductive formal verification. You simply can’t write better code.

Features_

## Key Advantages of SPARK

[ Learn SPARK Today ][1]
**
Foundations of the Ada Language

SPARK is based on the Ada programming language. All of the benefits of the Ada language translates
directly into benefits for SPARK. Wherever you can use Ada, from large server to small devices, you
can use SPARK.

[ Ada Benefits ][2]
**
Absence of Runtime Errors

Run-Time errors include issues traditionally found by automatic checks at run-time and defensive
code, such as out-of-bound array access, division by zero, overflow and more. SPARK proves that no
run-time errors are possible in your code.

**
Memory Safety

Through a combination of mitigation of dynamic memory usage, borrow-checking analysis and advanced
formal proof, SPARK formally demonstrates absence of memory issues such as use after free, access to
uninitialized memory or memory leaks and corruption.

**
Functional Safety

SPARK contracts - preconditions, postconditions, type invariants, assertions, etc. - are part of the
language, not structured comments. The contracts are checked by the compiler, proved by SPARK, and
have execution semantics.

**
No Dynamic Checks or Defensive Code

SPARK proves that defensive code and other run-time checks that may be inserted in the code will
never fail. This allows the compiler to remove them from the compiled code, ensuring optimal
efficiency while retaining guarantees of integrity.

**
No Undefined Behavior

SPARK is a dialect of Ada that removes Ada features leading to undefined behavior and extends Ada's
support for formal specification. When you develop in SPARK, you're using the safest, most secure
Ada possible.

**
Strong Typing

SPARK is a strongly typed language. Building upon Ada, types in SPARK are fundamental elements of
the software design. They are named, associated with properties, and checked for consistency
statically, via proof. When leveraged, SPARK’s type safety prevents errors that arise when mixing up
variables or accidentally converting between type representations.

**
Modularity

SPARK is fundamentally modular. SPARK performs deductive formal verification one subprogram at a
time. Called subprograms are assumed to satisfy their postconditions when their preconditions are
met. This ensures that SPARK scales and works well in teams. It also lets you use FFI without
sacrificing the ability to prove the correctness of your SPARK code.

[SPARK code snippet 3]
Mathematical Assurance_

## Build with Formal Methods

Formal methods represent software semantics using mathematics precisely, allowing analysis of
software behaviors prior to runtime. SPARK provides deductive formal verification: subprograms are
represented as Hoare triples, and SPARK proves that when the subprogram preconditions are met,
subprogram execution ensures the postcondition. SPARK proofs are sound - no false negatives, and
precise - no false positives, provided timeout is not reached. Since SPARK operates one subprogram
at a time, its analysis is scalable. Moreover, you can focus SPARK on only those parts of your
software that are of greatest concern.

[Binary code]
Related Products_

## Our SPARK Technology

[ See Our Technology ][3]
[

### GNAT Pro

GNAT Pro offers software development environments and toolchains for all versions of Ada across a
rich set of platforms, featuring IDEs, native and cross compilers, a multi-language build system,
multi-language debuggers, and configurable runtime libraries.

Explore
][4] [

### GNAT Static Analysis Suite

A comprehensive set of tools for enforcing coding standards, analyzing code metrics, and detecting
defects and vulnerabilities in Ada code.

Explore
][5] [

### GNAT Dynamic Analysis Suite

Provides unit testing and fuzz testing capabilities, along with a structural coverage analysis tool.

Explore
][6] [

### SPARK Pro

Leverages formal methods to automatically prevent or detect a wide range of bugs in Ada, ensuring
higher reliability and security.

Explore
][7]
Expert Support & Training_

## Our SPARK Services

AdaCore’s services help teams adopt SPARK with confidence, offering training, mentorship, and
certification support tailored to high-integrity projects.

[ See Our Services ][8]
[

### Certification support

Since SPARK is based on Ada, library & runtime certification and tool qualification are available
for DO-178, EN-50128, ISO 26262, ECSS-E-ST-40C / ECSS-Q-ST-80C and IEC 61508.

Learn More
][9] [

### Mentorship

SPARK represents a new programming paradigm. Our mentorship programs ensure your teams are ready to
get full value from adopting SPARK Pro.

Learn more
][10] [

### SPARK Language Training

AdaCore offers tailored training designed to help your team get up and running with SPARK.

Learn More
][11]
[Spark code snippet 2]
Where SPARK is Used Today_

## When Failure is not an Option

SPARK is playing an increasingly critical role in the development of safety- and security-critical
systems — from advanced defense applications and air-traffic management to firmware in medical and
industrial automation domains — because its design enforces the elimination of runtime errors,
ensures information-flow integrity, and enables formal proof of functional correctness. It is the
only technology available today that enables the deployment of formal methods directly on source
code at industrial scale. As the demand for mathematically assured software grows, SPARK continues
to expand its relevance across industries that tolerate nothing less than provable correctness.

[Binary code]
Get in Touch_

## Expert Guidance

Ready to bring the benefits of formal verification to your project? Our team can help you get
started with the right tools, training, and support.

[ Speak to an Expert ][12]
[4 222559 adacore nvidia case study cover image 568x800px]
Case Study_

## NVIDIA: Adoption of SPARK Ushers in a New Era in Security-Critical Software Development

Learn why NVIDIA made the “heretical” decision to abandon C/C++ and adopt SPARK as its coding
language of choice for security-critical software and firmware components.

[ Download ][13]
[Code]
[Nvidia logo svg]

> SPARK has a capability that is not found in most other programming languages. That is the ability
> to specify program requirements within the code itself and use the associated set of tools to
> ensure that the implementation matches its requirements. Essentially you are proving your programs
> are correct. That’s a very powerful capability.

**NVIDIA** Dhawal Kumar, Principle Software Engineer
Explore More_

## Latest News and Resources

[[2021 nvidia corporate key visual 16x9 dark 1080p]
[Blog Post]

Fabien Chouteau

## NVIDIA Security Team: “What if we just stopped using C?”

][14][[Welch allyn ada spark casestudy]
[Case Study]

## Ada and SPARK: Beyond Static Analysis for Medical Devices

][15][[245329 adacore code snippets 2]
[Blog Post]

Quentin Ochem

## Should I choose Ada, SPARK, or Rust over C/C++?

][16]

[1]: https://learn.adacore.com
[2]: https://www.adacore.com/languages/ada
[3]: https://www.adacore.com/offerings
[4]: https://www.adacore.com/gnat-pro-for-ada
[5]: https://www.adacore.com/static-analysis-suite
[6]: https://www.adacore.com/dynamic-analysis-suite
[7]: https://www.adacore.com/sparkpro
[8]: https://www.adacore.com/services
[9]: https://www.adacore.com/safety-security-certification
[10]: https://www.adacore.com/mentorship
[11]: https://www.adacore.com/enterprise-training
[12]: https://www.adacore.com/contact
[13]: https://www.adacore.com/case-studies/nvidia-adoption-of-spark-new-era-in-security-critical-sof
tware-development
[14]: https://www.adacore.com/blog/nvidia-security-team-what-if-we-just-stopped-using-c
[15]: https://www.adacore.com/case-studies/ada-and-spark-at-welch-allyn
[16]: https://www.adacore.com/blog/should-i-choose-ada-spark-or-rust-over-c-c
