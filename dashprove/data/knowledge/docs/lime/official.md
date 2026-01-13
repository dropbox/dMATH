# lime

[[Build Status]][1] [[Binder]][2]

This project is about explaining what machine learning classifiers (or models) are doing. At the
moment, we support explaining individual predictions for text classifiers or classifiers that act on
tables (numpy arrays of numerical or categorical data) or images, with a package called lime (short
for local interpretable model-agnostic explanations). Lime is based on the work presented in [this
paper][3] ([bibtex here for citation][4]). Here is a link to the promo video:

[[KDD promo video]][5]

Our plan is to add more packages that help users understand and interact meaningfully with machine
learning.

Lime is able to explain any black box classifier, with two or more classes. All we require is that
the classifier implements a function that takes in raw text or a numpy array and outputs a
probability for each class. Support for scikit-learn classifiers is built-in.

## Installation

The lime package is on [PyPI][6]. Simply run:

pip install lime

Or clone the repository and run:

pip install .

We dropped python2 support in `0.2.0`, `0.1.1.37` was the last version before that.

## Screenshots

Below are some screenshots of lime explanations. These are generated in html, and can be easily
produced and embedded in ipython notebooks. We also support visualizations using matplotlib,
although they don't look as nice as these ones.

#### Two class case, text

Negative (blue) words indicate atheism, while positive (orange) words indicate christian. The way to
interpret the weights by applying them to the prediction probabilities. For example, if we remove
the words Host and NNTP from the document, we expect the classifier to predict atheism with
probability 0.58 - 0.14 - 0.11 = 0.31.

[[twoclass]][7]

#### Multiclass case

[[multiclass]][8]

#### Tabular data

[[tabular]][9]

#### Images (explaining prediction of 'Cat' in pros and cons)

## Tutorials and API

For example usage for text classifiers, take a look at the following two tutorials (generated from
ipython notebooks):

* [Basic usage, two class. We explain random forest classifiers.][10]
* [Multiclass case][11]

For classifiers that use numerical or categorical data, take a look at the following tutorial (this
is newer, so please let me know if you find something wrong):

* [Tabular data][12]
* [Tabular data with H2O models][13]
* [Latin Hypercube Sampling][14]

For image classifiers:

* [Images - basic][15]
* [Images - Faces][16]
* [Images with Keras][17]
* [MNIST with random forests][18]
* [Images with PyTorch][19]

For regression:

* [Simple regression][20]

Submodular Pick:

* [Submodular Pick][21]

The raw (non-html) notebooks for these tutorials are available [here][22].

The API reference is available [here][23].

## What are explanations?

Intuitively, an explanation is a local linear approximation of the model's behaviour. While the
model may be very complex globally, it is easier to approximate it around the vicinity of a
particular instance. While treating the model as a black box, we perturb the instance we want to
explain and learn a sparse linear model around it, as an explanation. The figure below illustrates
the intuition for this procedure. The model's decision function is represented by the blue/pink
background, and is clearly nonlinear. The bright red cross is the instance being explained (let's
call it X). We sample instances around X, and weight them according to their proximity to X (weight
here is indicated by size). We then learn a linear model (dashed line) that approximates the model
well in the vicinity of X, but not necessarily globally. For more information, [read our paper][24],
or take a look at [this blog post][25].

## Contributing

Please read [this][26].

[1]: https://travis-ci.org/marcotcr/lime
[2]: https://mybinder.org/v2/gh/marcotcr/lime/master
[3]: https://arxiv.org/abs/1602.04938
[4]: https://github.com/marcotcr/lime/blob/master/citation.bib
[5]: https://www.youtube.com/watch?v=hUnRCxnydCc
[6]: https://pypi.python.org/pypi/lime
[7]: /marcotcr/lime/blob/master/doc/images/twoclass.png
[8]: /marcotcr/lime/blob/master/doc/images/multiclass.png
[9]: /marcotcr/lime/blob/master/doc/images/tabular.png
[10]: https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.htm
l
[11]: https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html
[12]: https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20feat
ures.html
[13]: https://marcotcr.github.io/lime/tutorials/Tutorial_H2O_continuous_and_cat.html
[14]: /marcotcr/lime/blob/master/doc/notebooks/Latin%20Hypercube%20Sampling.ipynb
[15]: https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html
[16]: https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Faces%20and%20GradBo
ost.ipynb
[17]: https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classificati
on%20Keras.ipynb
[18]: https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20MNIST%20and%20RF.ipy
nb
[19]: https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch
.ipynb
[20]: https://marcotcr.github.io/lime/tutorials/Using%2Blime%2Bfor%2Bregression.html
[21]: https://github.com/marcotcr/lime/tree/master/doc/notebooks/Submodular%20Pick%20examples.ipynb
[22]: https://github.com/marcotcr/lime/tree/master/doc/notebooks
[23]: https://lime-ml.readthedocs.io/en/latest/
[24]: https://arxiv.org/abs/1602.04938
[25]: https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanatio
ns-lime
[26]: /marcotcr/lime/blob/master/CONTRIBUTING.md
