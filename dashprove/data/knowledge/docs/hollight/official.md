# The HOL Light theorem prover

** Written by [John Harrison][1] drawing on the work of
[Mike Gordon][2] [Tom Melham][3] [Robin Milner][4] [Larry Paulson][5] [Konrad Slind][6]
and many other HOL and LCF researchers**

****

HOL Light is a computer program to help users prove interesting mathematical theorems completely
formally in higher order logic. It sets a very exacting standard of correctness, but provides a
number of automated tools and pre-proved mathematical theorems (e.g. about arithmetic, basic set
theory and real analysis) to save the user work. It is also fully programmable, so users can extend
it with new theorems and inference rules without compromising its soundness. There are a number of
versions of [HOL][7], going back to [Mike Gordon][8]'s work in the early 80s. Compared with other
HOL systems, HOL Light uses a much simpler logical core and has little legacy code, giving the
system a simple and uncluttered feel. Despite its simplicity, it offers theorem proving power
comparable to, and in some areas greater than, other versions of HOL, and has been used for some
significant industrial-scale verification applications.

**HOL Light is now hosted on Github** so you can get the sources from the [Github repository][9].
You can browse individual source files online, or check out all the code using [git][10]. For
example, the following command will copy all the code from the Github repository into a new
directory hol-light (assumed not to exist already):

git clone https://github.com/jrh13/hol-light.git
If you use a debian-based Linux distribution, then you can get a ready-to-use HOL Light system
together with useful auxiliary tools simply by installing the hol-light package (thanks to [Hendrik
Tews ][11]). For example
sudo apt-get install hol-light
Otherwise, you can always install it yourself. HOL Light is written in [Objective CAML][12] (OCaml),
and it should work with any reasonably recent version. To build with OCaml 3.10 and later you will
also need to install [camlp5][13], version 5.07 or higher. See the README file in the distribution
for detailed installation instructions.

The following lists some available documentation and resources:


* Reference Manual available as [online crosslinked HTML][14] or as [one PDF file][15] (also an
  [older variant][16] keyed to version 2.20).
* **[Tutorial][17]**, which tries to teach HOL Light through examples.
* Quick Reference Guide compiled by [Freek Wiedijk][18] ([text][19], [PDF][20], [Postscript][21],
  [DVI][22], [LaTeX][23])
* Jeremy Bem's [hol-online][24], and the earlier project [Formalpedia][25], use a processed version
  of HOL Light to produce an online repository of software and formal mathematics.
* Summary of many HOL source files, written by Carl Witty ([text][26])
* Old user manual, not fully updated from the older CAML Light version ([DVI][27], [Postscript][28]
  and [PDF][29]).
* The [hol-info][30] mailing list is a good place for HOL-related discussion
* [Josef Urban][31]'s [HOL Proof Advisor][32] uses automated data mining to produce proof advice.
Here are some applications of HOL Light:

* Formalization of floating-point arithmetic, and the formal verification of several floating-point
  algorithms at Intel.
  See this [paper][33] for a quick summary and more references, and [this one][34] for a more
  detailed presentation.
* The [Flyspeck project][35] to machine-check [Tom Hales][36]'s proof of the Kepler conjecture.
  Tom has already proved the Jordan Curve Theorem and other relevant results in HOL Light.
* Many other mathematical results of varying degrees of difficulty have been verified in HOL Light.
  See for example the HOL Light entries on the [Formalizing 100 Theorems][37] page.
HOL Light is free open source software. It comes with no warranty of any kind (see the LICENCE file
in the distribution), and no guarantee of maintainance. However, please feel free to send any
comments or questions to the author at . Last updated by [John Harrison][38] on Fri 13th January
2017.

[1]: http://www.cl.cam.ac.uk/~jrh13/
[2]: http://www.cl.cam.ac.uk/users/mjcg
[3]: http://web.comlab.ox.ac.uk/oucl/people/tom.melham.html
[4]: http://www.cl.cam.ac.uk/users/rm135
[5]: http://www.cl.cam.ac.uk/users/lcp
[6]: http://www.cs.utah.edu/~slind/
[7]: http://www.cl.cam.ac.uk/Research/HVG/HOL/
[8]: http://www.cl.cam.ac.uk/users/mjcg
[9]: https://github.com/jrh13/hol-light/
[10]: https://en.wikipedia.org/wiki/Git_%28software%29
[11]: http://askra.de/
[12]: http://caml.inria.fr/ocaml/index.en.html
[13]: https://camlp5.github.io
[14]: reference.html
[15]: reference.pdf
[16]: reference_220.pdf
[17]: tutorial.pdf
[18]: http://www.cs.ru.nl/~freek/
[19]: holchart.txt
[20]: holchart.pdf
[21]: holchart.ps
[22]: holchart.dvi
[23]: holchart.tex
[24]: http://code.google.com/p/hol-online/
[25]: http://formalpedia.org/
[26]: summary.txt
[27]: manual-1.1.dvi.gz
[28]: manual-1.1.ps.gz
[29]: http://www.cl.cam.ac.uk/~jrh13/hol-light/manual-1.1.pdf
[30]: http://lists.sourceforge.net/lists/listinfo/hol-info
[31]: http://kti.ms.mff.cuni.cz/~urban/
[32]: http://lipa.ms.mff.cuni.cz/~urban/holpademo.html
[33]: http://www.cl.cam.ac.uk/~jrh13/papers/iday.html
[34]: http://www.cl.cam.ac.uk/~jrh13/papers/sfm.html
[35]: http://code.google.com/p/flyspeck/
[36]: http://www.math.pitt.edu/~thales/
[37]: http://www.cs.ru.nl/~freek/100/
[38]: http://www.cl.cam.ac.uk/~jrh13
