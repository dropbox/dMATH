# Getting Started[#][1]

## Installation

Interpret is supported across Windows, Mac and Linux on Python 3.5+

pip

pip install interpret

conda

conda install -c conda-forge interpret

source

git clone [interpretml/interpret.git][2] && cd interpret/scripts && make install

InterpretML supports training interpretable models (**glassbox**), as well as explaining existing ML
pipelines (**blackbox**). Let’s walk through an example of each using the UCI adult income
classification dataset.

## Download and Prepare Data

First, we will load the data into a standard pandas dataframe or a numpy array, and create a train /
test split. There’s no special preprocessing necessary to use your data with InterpretML.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None)
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]
X = df.iloc[:, :-1]
y = (df.iloc[:, -1] == " >50K").astype(int)

seed = 42
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

## Train a Glassbox Model

**Glassbox** models are designed to be completely interpretable, and often provide similar accuracy
to state-of-the-art methods.

InterpretML lets you train many of the latest glassbox models with the familiar scikit-learn
interface.

from interpret.glassbox import ExplainableBoostingClassifier
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
ExplainableBoostingClassifier()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the
notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with
nbviewer.org.
ExplainableBoostingClassifier
iFitted
ExplainableBoostingClassifier()

## Explain the Glassbox

Glassbox models can provide explanations on a both global (overall behavior) and local (individual
predictions) level.

**Global explanations** are useful for understanding what a model finds important, as well as
identifying potential flaws in its decision making (i.e. racial bias).

The inline visualization embedded here are exactly what gets produced in the notebook.

For this global explanation, the initial summary page shows the most important features overall. You
can use the dropdown to search, filter, and select individual features to drill down deeper into.

Try looking at the “Age” feature to see how the probability of high income varies with Age, or the
“Race” or “Gender” features to observe potential bias the model may have learned.

from interpret import show
show(ebm.explain_global())





**Local explanations** show how a single prediction is made. For glassbox models, these explanations
are exact – they perfectly describe how the model made its decision.

These explanations are useful for describing to end users which factors were most influential for a
prediction. In the local explanation below for instance “2”, the probability of high income was
0.93, largely due to having a high value for the CapitalGains feature.

The values shown here are **log-odds** scores from the EBM, which are added and passed through a
logistic-link function to get the final prediction, just like logistic regression.

show(ebm.explain_local(X_test[:5], y_test[:5]), 0)





## Build a Blackbox Pipeline

**Blackbox interpretability** methods can extract explanations from any machine learning pipeline.
This includes model ensembles, pre-processing steps, and complex models such as deep neural nets.

Let’s start by training a random forest that is first pre-processed with principal component
analysis.

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# We have to transform categorical variables to use sklearn models
X = pd.get_dummies(X, prefix_sep='.').astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

#Blackbox system can include preprocessing, not just a classifier!
pca = PCA()
rf = RandomForestClassifier(random_state=seed)

blackbox_model = Pipeline([('pca', pca), ('rf', rf)])
blackbox_model.fit(X_train, y_train)
Pipeline(steps=[('pca', PCA()),
                ('rf', RandomForestClassifier(random_state=42))])
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the
notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with
nbviewer.org.
Pipeline
[?Documentation for Pipeline][3]iFitted
Pipeline(steps=[('pca', PCA()),
                ('rf', RandomForestClassifier(random_state=42))])
PCA
[?Documentation for PCA][4]
PCA()
RandomForestClassifier
[?Documentation for RandomForestClassifier][5]
RandomForestClassifier(random_state=42)

## Explain the Blackbox

All you need for a blackbox interpretability method is a predict function from the target ML
pipeline.

Blackbox interpretability methods generally work by perturbing input data repeatedly passing it
through the pipeline, and observing how the final prediction changes.

As a result both global and local explanations are approximate, and may sometimes be inaccurate. Be
cautious of the results in high-stakes environments.

from interpret.blackbox import LimeTabular
from interpret import show

lime = LimeTabular(blackbox_model, X_train, random_state=seed)
show(lime.explain_local(X_test[:5], y_test[:5]), 0)

[1]: #getting-started
[2]: https://github.com/interpretml/interpret.git
[3]: https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html
[4]: https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html
[5]: https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html
