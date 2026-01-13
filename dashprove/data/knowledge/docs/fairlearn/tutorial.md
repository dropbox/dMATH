# Get Started[#][1]

## Installation[#][2]

Fairlearn can be installed with `pip` from [PyPI][3] as follows:

pip install fairlearn

Fairlearn is also available on [conda-forge][4]:

conda install -c conda-forge fairlearn

For further information on how to install Fairlearn and its optional dependencies, please check out
the [Installation Guide][5].

If you are updating from a previous version of Fairlearn, please see [Version guide][6].

Note

The Fairlearn API is still evolving, so example code in this documentation may not work with every
version of Fairlearn. Please use the version selector to get to the instructions for the appropriate
version. The instructions for the `main` branch require Fairlearn to be installed from a clone of
the repository.

## Overview of Fairlearn[#][7]

The Fairlearn package has two components:

* *Metrics* for assessing which groups are negatively impacted by a model, and for comparing
  multiple models in terms of various fairness and accuracy metrics.
* *Algorithms* for mitigating unfairness in a variety of AI tasks and along a variety of fairness
  definitions.

## Fairlearn in 10 minutes[#][8]

The Fairlearn toolkit can assist in assessing and mitigation unfairness in Machine Learning models.
It’s impossible to provide a sufficient overview of fairness in ML in this Quickstart tutorial, so
we highly recommend starting with our [User Guide][9]. Fairness is a fundamentally sociotechnical
challenge and cannot be solved with technical tools alone. They may be helpful for certain tasks
such as assessing unfairness through various metrics, or to mitigate observed unfairness when
training a model. Additionally, fairness has different definitions in different contexts and it may
not be possible to represent it quantitatively at all.

Given these considerations this Quickstart tutorial merely provides short code snippet examples of
how to use basic Fairlearn functionality for those who are already intimately familiar with fairness
in ML. The example below is about binary classification, but we similarly support regression.

### Prerequisites[#][10]

In order to run the code samples in the Quickstart tutorial, you need to install the following
dependencies:

pip install fairlearn matplotlib

### Loading the dataset[#][11]

For this example, we use a [clinical dataset][12] of hospital re-admissions over a ten-year period
(1998-2008) for diabetic patients across 130 different hospitals in the U.S. This scenario builds
upon prior research on how racial disparities impact health care resource allocation in the U.S. For
an in-depth analysis of this dataset, review the [SciPy tutorial][13] that the Fairlearn team
presented in 2021.

We will use machine learning to predict whether an individual in the dataset is readmitted to the
hospital within 30 days of hospital release. A hospital readmission within 30 days can be viewed as
a proxy that the patients needed more assistance at the release time. In the next section, we build
a classification model to accomplish the prediction task.

>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> from fairlearn.datasets import fetch_diabetes_hospital
>>> data = fetch_diabetes_hospital(as_frame=True)
>>> X = data.data.copy()
>>> X.drop(columns=["readmitted", "readmit_binary"], inplace=True)
>>> y = data.target
>>> X_ohe = pd.get_dummies(X)
>>> race = X['race']
>>> race.value_counts()
race
Caucasian          76099
AfricanAmerican    19210
Unknown             2273
Hispanic            2037
Other               1506
Asian                641
Name: count, dtype: int64
[[_images/sphx_glr_plot_quickstart_counts_001.png] ][14]

### Evaluating fairness-related metrics[#][15]

First, Fairlearn provides fairness-related metrics that can be compared between groups and for the
overall population. Using existing metric definitions from [scikit-learn][16] we can evaluate
metrics for subgroups within the data as below:

>>> from fairlearn.metrics import MetricFrame
>>> from sklearn.metrics import accuracy_score, balanced_accuracy_score
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.model_selection import train_test_split
>>> np.random.seed(42)  # set seed for consistent results
>>> X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X_ohe, y, race, random_stat
e=123)
>>> classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
>>> classifier.fit(X_train, y_train)
DecisionTreeClassifier(...)
>>> y_pred = (classifier.predict_proba(X_test)[:,1] >= 0.1)
>>> mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=A_test
)
>>> mf.overall.item()
0.514...
>>> mf.by_group
race
AfricanAmerican    0.530935
Asian              0.658683
Caucasian          0.503535
Hispanic           0.612524
Other              0.591549
Unknown            0.574576
Name: accuracy_score, dtype: float64

Note that our decision threshold for positive predictions is 0.1. In practice, this threshold would
be driven by risk or capacity considerations. For this example, we set the threshold based on the
risk of readmission. The threshold of 0.1 corresponds to saying that a 10% risk of readmission is
viewed as sufficient for referral to a post-discharge care program. Fairlearn has many standard
metrics built-in, such as false negative rate, i.e., the rate of occurrence of negative
classifications when the true value of the outcome label is positive. In the context of this
dataset, the false positive rate captures the individuals who in reality would be readmitted to the
hospital, but the model does not predict that outcome.

>>> from fairlearn.metrics import false_negative_rate
>>> mf = MetricFrame(metrics=false_negative_rate, y_true=y_test, y_pred=y_pred, sensitive_features=A
_test)
>>> mf.overall.item()
0.309...
>>> mf.by_group
race
AfricanAmerican    0.296089
Asian              0.500000
Caucasian          0.308555
Hispanic           0.307692
Other              0.333333
Unknown            0.420000
Name: false_negative_rate, dtype: float64

Fairlearn also allows us to quickly plot these metrics from the
[`fairlearn.metrics.MetricFrame`][17]

metrics = {
    "accuracy": accuracy_score,
    "precision": zero_div_precision_score,
    "false positive rate": false_positive_rate,
    "false negative rate": false_negative_rate,
    "selection rate": selection_rate,
    "count": count,
}
metric_frame = MetricFrame(
    metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=A_test
)
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)

[[_images/sphx_glr_plot_quickstart_001.png] ][18]

### Mitigating disparity[#][19]

If we observe disparities between groups we may want to create a new model while specifying an
appropriate fairness constraint. Note that the choice of fairness constraints is crucial for the
resulting model, and varies based on application context. Since both false positives and false
negatives are relevant for fairness in this hypothetical example, we can attempt to mitigate the
observed disparity using the fairness constraint called Equalized Odds, which bounds disparities in
both types of error. In real world applications we need to be mindful of the sociotechnical context
when making such decisions. The Exponentiated Gradient mitigation technique used fits the provided
classifier using Equalized Odds as the constraint and a suitably weighted Error Rate as the
objective, leading to a vastly reduced difference in accuracy:

>>> from fairlearn.reductions import ErrorRate, EqualizedOdds, ExponentiatedGradient
>>> objective = ErrorRate(costs={'fp': 0.1, 'fn': 0.9})
>>> constraint = EqualizedOdds(difference_bound=0.01)
>>> classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
>>> mitigator = ExponentiatedGradient(classifier, constraint, objective=objective)
>>> mitigator.fit(X_train, y_train, sensitive_features=A_train)
ExponentiatedGradient(...)
>>> y_pred_mitigated = mitigator.predict(X_test)
>>> mf_mitigated = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred_mitigated, sensi
tive_features=A_test)
>>> mf_mitigated.overall.item()
0.5251...
>>> mf_mitigated.by_group
race
AfricanAmerican    0.524358
Asian              0.562874
Caucasian          0.525588
Hispanic           0.549902
Other              0.478873
Unknown            0.511864
Name: accuracy_score, dtype: float64

Note that [`ExponentiatedGradient`][20] does not have a predict_proba method, but we can adjust the
target decision threshold by specifying (possibly unequal) costs for false positives and false
negatives. In our example we use the cost of 0.1 for false positives and 0.9 for false negatives.
Without fairness constraints, this would exactly correspond to referring patients with the
readmission risk of 10% or higher (as we used earlier).

## What’s next?[#][21]

Please refer to our [User Guide][22] for a comprehensive view on Fairness in Machine Learning and
how Fairlearn fits in, as well as an exhaustive guide on all parts of the toolkit. For concrete
examples check out the [Example Notebooks][23] section. Finally, we also have a collection of
[Frequently asked questions][24].

[1]: #get-started
[2]: #installation
[3]: https://pypi.org/project/fairlearn
[4]: https://anaconda.org/conda-forge/fairlearn
[5]: user_guide/installation_and_version_guide/installation_guide.html#installation-guide
[6]: user_guide/installation_and_version_guide/version_guide.html#version-guide
[7]: #overview-of-fairlearn
[8]: #fairlearn-in-10-minutes
[9]: user_guide/index.html#user-guide
[10]: #prerequisites
[11]: #loading-the-dataset
[12]: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
[13]: https://github.com/fairlearn/talks/tree/main/2021_scipy_tutorial
[14]: auto_examples/plot_quickstart_counts.html
[15]: #evaluating-fairness-related-metrics
[16]: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
[17]: api_reference/generated/fairlearn.metrics.MetricFrame.html#fairlearn.metrics.MetricFrame
[18]: auto_examples/plot_quickstart.html
[19]: #mitigating-disparity
[20]: api_reference/generated/fairlearn.reductions.ExponentiatedGradient.html#fairlearn.reductions.E
xponentiatedGradient
[21]: #what-s-next
[22]: user_guide/index.html#user-guide
[23]: auto_examples/index.html#examples
[24]: faq.html#faq
