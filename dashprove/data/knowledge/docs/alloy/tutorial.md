# Tutorial for Alloy Analyzer 4.0

[*Alloy*][1] is a lightweight modelling language for software design. It is amenable to a fully
automatic analysis, using the Alloy Analyzer, and provides a visualizer for making sense of
solutions and counterexamples it finds.

This introduction to *Alloy* includes:

* a tutorial for writing declarative models in *Alloy*
* a walkthrough of the *Alloy* interface and visualizer
* tips and techniques to help you with the modeling process
* two thoroughly explained sample models which are developed as running examples

This tutorial does not:

* provide an exhaustive list of *Alloy* commands, although many are described in sidenotes;
* focus on the technical semantics of *Alloy*, although we do touch upon it in several sidenotes.
  Rather, we focus on getting you to a point where you can read, write, and analyze reasonably
  advanced models;
* give a lengthy description of the background philosophy behind modeling and declarative modeling.
  The philosophy chapter gives a summary of these issues, but the focus of this tutorial is to teach
  *Alloy* not convincing you to use it.

## Instructions for using this tutorial

The main text of the tutorial will appear in this frame of the window. We will use running sample
models, displayed in the top-right frame. Relevant side notes will be loaded in the bottom-right
frame. Sidenotes provide supplemental information and can be skipped without impeding your
understanding of the current model.

## The Chapters (start here)

* [Chapter 0][2], gives background for declarative modeling, compares Alloy to its alternatives, and
  further introduces the tutorial.
  
* [Chapter 1][3] follows the development and analysis of a simple *Alloy* model of a file system,
  This chapter includes an [ *Alloy* interface walkthrough][4]. This will also introduce you to
  Alloy's improved solution visualizer.
  
* [Chapter 2][5] examines a more advanced *Alloy* model which solves the famous [ River Crossing][6]
  planning problem, using the util/ordering module for modeling ordered state.

*Alloy* was developed at [ MIT][7] by the [ Software Design Group][8] under the guidance of [ Daniel
Jackson][9]. This research was funded by grant *0086154* from the [ ITR][10] program of the [
National Science Foundation][11], by a grant from [NASA][12], and by an endowment from Doug and Pat
Ross.

This tutorial was originally written by Rob Seater and Greg Dennis. It has since been updated to
match the new Alloy4 syntax by Daniel Le Berre and Felix Chang.

[1]: https://alloytools.org/
[2]: frame-FS-0.html
[3]: frame-FS-1.html
[4]: frame-FS-2.html
[5]: frame-RC-1.html
[6]: sidenote-RC-puzzle.html
[7]: http://web.mit.edu/
[8]: http://sdg.csail.mit.edu/
[9]: http://sdg.csail.mit.edu/people/dnj.html
[10]: http://www.itr.nsf.gov/
[11]: http://www.nsf.gov/
[12]: http://www.nasa.gov/
