[Frama-C][1]

* [Features][2]
* [Documentation][3]
* [Publications][4]
* [Blog][5]
* [Jobs][6]
* [Contact][7]
* [Download][8]
[******][9]
[Plugins][10] [Kernel][11] [Specification][12] [Ivette (GUI)][13]

# ANSI/ISO C Specification Language

## Quick description

The ANSI/ISO C Specification Langage (ACSL) is a behavioral specification language for C programs.
The design of ACSL is inspired of [JML][14]. It also inherits a lot from the specification language
of the source code analyzer Caduceus, a previous development of one of the partners in the Frama-C
project.

ACSL can express a wide range of functional properties. The paramount notion in ACSL is the function
contract. While many software engineering experts advocate the "function contract mindset" when
designing complex software, they generally leave the actual expression of the contract to run-time
assertions, or to comments in the source code. ACSL is expressly designed for writing the kind of
properties that make up a function contract. ACSL is a *formal* language.

#include <stddef.h>
/*@
  requires \valid(a+(0..n-1));1ACSL provides specification primitives to cover the low-level aspects
 of the C programming language

  assigns  a[0..n-1];

  ensures2As a formal language, ACSL enables a precise specification of function contracts. That mak
es the specification not only understandable by a human, but also manipulable by an analyzer. Furthe
rmore, as a complete specification is not always useful, the contract can be partial, it depends on 
what one wants to verify.
  \forall integer i;
    0 <= i < n ==> a[i] == 0;
*/
void set_to_0(int* a, size_t n){
  size_t i;

  /*@
    loop invariant 0 <= i <= n;
    loop invariant
    \forall integer j;3It also allows more abstract reasoning through mathematical or logic types, o
r through the definition of high level ideas, like "the function expects a valid linked list".
      0 <= j < i ==> a[j] == 0;
    loop assigns i, a[0..n-1];
    loop variant n-i;
  */
  for(i = 0; i < n; ++i)
    a[i] = 0;
}
ACSL provides specification primitives to cover the low-level aspects of the C programming language
As a formal language, ACSL enables a precise specification of function contracts. That makes the
specification not only understandable by a human, but also manipulable by an analyzer. Furthermore,
as a complete specification is not always useful, the contract can be partial, it depends on what
one wants to verify.
It also allows more abstract reasoning through mathematical or logic types, or through the
definition of high level ideas, like "the function expects a valid linked list".

[WP][15] and the older [Jessie][16] plug-ins use Hoare-style weakest precondition computations to
formally prove ACSL properties. The process can be quite automatic, thanks to external theorem
provers such as Alt-Ergo, Z3 or CVC4, or more interactive, with WP's built-in interactive prover or
the use of the Coq proof assistant. Other plug-ins, such as the [Eva][17] plug-in, may also
contribute to the verification of ACSL properties. They may also report static analysis results in
terms of new asserted ACSL properties inside the source code.

## More information

* The ACSL manual has its own [repository][18], together with ACSL++ (its companion language for
  specifying C++ programs and analyzing them with the [frama-clang][19] plug-in. Pdf versions of the
  manual are available on the [ release page][20].
* A comprehensive tutorial on [the WP plugin][21] and ACSL specifications is available [ here][22].
* Another tutorial, with specifications inspired notably from C++ containers is available [
  here][23].

## Manuals

* [ACSL implementation in the latest Frama-C release][24]
* [ACSL v1.23][25] - Germanium release
* [ACSL v1.22][26] - Gallium release
* [ACSL v1.21][27] - Zinc release
* [ACSL v1.20][28] - Nickel, and Copper releases
* [ACSL v1.19][29] - Cobalt release
* [ACSL v1.18][30] - Manganese, and Iron releases
* [ACSL v1.17][31] - Vanadium, and Chromium releases
* [ACSL v1.16][32] - Titanium release
* [ACSL v1.15][33] - Scandium release
* [ACSL v1.14][34] - Argon, Potassium, and Calcium releases
* [ACSL v1.13][35] - Chlorine release
* [ACSL v1.12][36] - Silicon, Phosphorus, and Sulfur releases
* [ACSL v1.11][37] - Aluminium release
* [ACSL v1.10][38] - Magnesium release
* [ACSL v1.9][39] - Sodium release
* [ACSL v1.8][40] - Neon release
* [ACSL v1.7][41] - Fluorine release
* [ACSL v1.6][42] - Oxygen release
* [ACSL v1.5][43] - Carbon, and Nitrogen releases
* [ACSL v1.4][44] - Lithium, Beryllium, and Boron releases
* [ACSL v1.3][45] - Helium release
* [ACSL v1.2][46] - Hydrogen release
Copyright © 2007-2025 Frama-C. All Rights Reserved.

* [Terms Of Use][47]
* [Authors][48]
* [Acknowledgements][49]

[1]: /index.html
[2]: /html/kernel-plugin.html
[3]: /html/documentation.html
[4]: /html/publications.html
[5]: /blog/index.html
[6]: /html/jobs.html
[7]: /html/contact.html
[8]: /html/get-frama-c.html
[9]: /html/get-frama-c.html
[10]: /html/kernel-plugin.html
[11]: /html/kernel.html
[12]: /html/acsl.html
[13]: /html/ivette.html
[14]: http://www.eecs.ucf.edu/~leavens/JML/index.shtml
[15]: ../fc-plugins/wp.html
[16]: ../fc-plugins/jessie.html
[17]: ../fc-plugins/eva.html
[18]: https://github.com/acsl-language/acsl/
[19]: ../fc-plugins/frama-clang.html
[20]: https://github.com/acsl-language/acsl/releases
[21]: ../fc-plugins/wp.html
[22]: https://allan-blanchard.fr/frama-c-wp-tutorial.html
[23]: https://github.com/fraunhoferfokus/acsl-by-example
[24]: /download/frama-c-acsl-implementation.pdf
[25]: /download/acsl-1.23.pdf
[26]: /download/acsl-1.22.pdf
[27]: /download/acsl-1.21.pdf
[28]: /download/acsl-1.20.pdf
[29]: /download/acsl-1.19.pdf
[30]: /download/acsl-1.18.pdf
[31]: /download/acsl-1.17.pdf
[32]: /download/acsl-1.16.pdf
[33]: /download/acsl-1.15.pdf
[34]: /download/acsl-1.14.pdf
[35]: /download/acsl_1.13.pdf
[36]: /download/acsl_1.12.pdf
[37]: /download/acsl_1.11.pdf
[38]: /download/acsl_1.10.pdf
[39]: /download/acsl_1.9.pdf
[40]: /download/acsl_1.8.pdf
[41]: /download/acsl_1.7.pdf
[42]: /download/acsl_1.6.pdf
[43]: /download/acsl_1.5.pdf
[44]: /download/acsl_1.4.pdf
[45]: /download/acsl_1.3.pdf
[46]: /download/acsl_1.2.pdf
[47]: /html/terms-of-use.html
[48]: /html/authors.html
[49]: /html/acknowledgement.html
