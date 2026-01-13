[Edit][1]

1. [Overview][2]

# Introduction

**Explainability provides us with algorithms that give insights into trained model predictions.** It
allows us to answer questions such as:

* How does a prediction **change** dependent on feature inputs?
* What features **are** or **are not** important for a given prediction to hold?
* What set of features would you have to minimally **change** to obtain a **new** prediction of your
  choosing?
* How does each feature **contribute** to a model's prediction?
[Model augmented with explainabilty]
Model Augmentation with explainability

Alibi provides a set of **algorithms** or **methods** known as **explainers**. Each explainer
provides some kind of insight about a model. The set of insights available given a trained model is
dependent on a number of factors. For instance, if the model is a [regression model][3] it makes
sense to ask how the prediction varies for some regressor. Whereas it doesn't make sense to ask what
minimal change is required to obtain a new class prediction. In general, given a model the
explainers available from **Alibi** are constrained by:

* The **type of data** the model handles. Each insight applies to some or all of the following kinds
  of data: image, tabular or textual.
* The **task the model** performs. Alibi provides explainers for regression or [classification][4]
  models.
* The **type of model** used. Examples of model types include [neural networks][5] and [random
  forests][6].

### Applications

As machine learning methods have become more complex and more mainstream, with many industries now
[incorporating AI][7] in some form or another, the need to understand the decisions made by models
is only increasing. Explainability has several applications of importance.

* **Trust:** At a core level, explainability builds [trust][8] in the machine learning systems we
  use. It allows us to justify their use in many contexts where an understanding of the basis of the
  decision is paramount. This is a common issue within machine learning in medicine, where acting on
  a model prediction may require expensive or risky procedures to be carried out.
* **Testing:** Explainability might be used to [audit financial models][9] that aid decisions about
  whether to grant customer loans. By computing the attribution of each feature towards the
  prediction the model makes, organisations can check that they are consistent with human
  decision-making. Similarly, explainability applied to a model trained on image data can explicitly
  show the model's focus when making decisions, aiding [debugging][10]. Practitioners must be wary
  of [misuse][11], however.
* **Functionality:** Insights can be used to augment model functionality. For instance, providing
  information on top of model predictions such as how to change model inputs to obtain desired
  outputs.
* **Research:** Explainability allows researchers to understand how and why opaque models make
  decisions. This can help them understand more broadly the effects of the particular model or
  training schema they're using.

### Black-box vs White-box methods

Some explainers apply only to specific types of models such as the [Tree SHAP][12] methods which can
only be used with [tree-based models][13]. This is the case when an explainer uses some aspect of
that model's internal structure. If the model is a neural network then some methods require taking
gradients of the model predictions with respect to the inputs. Methods that require access to the
model internals are known as **white-box** methods. Other explainers apply to any type of model.
They can do so because the underlying method doesn't make use of the model internals. Instead, they
only need to have access to the model outputs given particular inputs. Methods that apply in this
general setting are known as **black-box** methods. Typically, white-box methods are faster than
black-box methods as they can exploit the model internals. For a more detailed discussion see
[white-box and black-box models][14].

**Note 1: Black-box Definition** The use of black-box here varies subtly from the conventional use
within machine learning. In most other contexts a model is a black-box if the mechanism by which it
makes predictions is too complicated to be interpretable to a human. Here we use black-box to mean
that the explainer method doesn't need access to the model internals to be applied.

### Global and Local Insights

Insights can be categorised into two categories — local and global. Intuitively, a local insight
says something about a single prediction that a model makes. For example, given an image classified
as a cat by a model, a local insight might give the set of features (pixels) that need to stay the
same for that image to remain classified as a cat.

On the other hand, global insights refer to the behaviour of the model over a range of inputs. As an
example, a plot that shows how a regression prediction varies for a given feature. These insights
provide a more general understanding of the relationship between inputs and model predictions.

[Local and Global insights]
Local and Global insights

### Biases

The explanations Alibi's methods provide depend on the model, the data, and — for local methods —
the instance of interest. Thus Alibi allows us to obtain insight into the model and, therefore, also
the data, albeit indirectly. There are several pitfalls of which the practitioner must be wary.

Often bias exists in the data we feed machine learning models even when we exclude sensitive
factors. Ostensibly explainability is a solution to this problem as it allows us to understand the
model's decisions to check if they're appropriate. However, human bias itself is still an element.
Hence, if the model is doing what we expect it to on biased data, we are vulnerable to using
explainability to justify relations in the data that may not be accurate. Consider:

> *"Before launching the model, risk analysts are asked to review the Shapley value explanations to
> ensure that the model exhibits expected behavior (i.e., the model uses the same features that a
> human would for the same task)."* — [*Explainable Machine Learning in Deployment*][15]

The critical point here is that the risk analysts in the above scenario must be aware of their own
bias and potential bias in the dataset. The Shapley value explanations themselves don't remove this
source of human error; they just make the model less opaque.

Machine learning engineers may also have expectations about how the model should be working. An
explanation that doesn't conform to their expectations may prompt them to erroneously decide that
the model is "incorrect". People usually expect classifiers trained on image datasets to use the
same structures humans naturally do when identifying the same classes. However, there is no reason
to believe such models should behave the same way we do.

Interpretability of insights can also mislead. Some insights such as [**anchors**][16] give
conditions for a classifiers prediction. Ideally, the set of these conditions would be small.
However, when obtaining anchors close to decision boundaries, we may get a complex set of conditions
to differentiate that instance from near members of a different class. Because this is harder to
understand, one might write the model off as incorrect, while in reality, the model performs as
desired.

## Types of Insights

Alibi provides several local and global insights with which to explore and understand models. The
following gives the practitioner an understanding of which explainers are suitable in which
situations.

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Accumulated Local Effects][17]

Global

Black-box

Classification, Regression

Tabular (numerical)

How does model prediction vary with respect to features of interest?

[docs][18], [paper][19]

[Partial Dependence][20]

Global

Black-box, White-box (*scikit-learn*)

Classification, Regression

Tabular (numerical, categorical)

How does model prediction vary with respect to features of interest?

[docs][21], [paper][22]

[Partial Dependence Variance][23]

Global

Black-box, White-box (*scikit-learn*)

Classification, Regression

Tabular (numerical, categorical)

Which are the most important features globally? How much do features interact globally?

[docs][24], [paper][25]

[Permutation importance][26]

Global

Black-box

Classification, Regression

Tabular (numerical, categorical)

Which are the most important features globally?

[docs][27], [paper][28]

[Anchors][29]

Local

Black-box

Classification

Tabular (numerical, categorical), Text and Image

Which set of features of a given instance are sufficient to ensure the prediction stays the same?

[docs][30], [paper][31]

[Pertinent Positives][32]

Local

Black-box, White-box (*TensorFlow*)

Classification

Tabular (numerical), Image

""

[docs][33], [paper][34]

[Integrated Gradients][35]

Local

White-box (*TensorFlow*)

Classification, Regression

Tabular (numerical, categorical), Text and Image

What does each feature contribute to the model prediction?

[docs][36], [paper][37]

[Kernel SHAP][38]

Local

Black-box

Classification, Regression

Tabular (numerical, categorical)

""

[docs][39], [paper][40]

[Tree SHAP (path-dependent)][41]

Local

White-box (*XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models*)

Classification, Regression

Tabular (numerical, categorical)

""

[docs][42], [paper][43]

[Tree SHAP (interventional)][44]

Local

White-box (*XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models*)

Classification, Regression

Tabular (numerical, categorical)

""

[docs][45], [paper][46]

[Counterfactual Instances][47]

Local

Black-box (*differentiable*), White-box (*TensorFlow*)

Classification

Tabular (numerical), Image

What minimal change to features is required to reclassify the current prediction?

[docs][48], [paper][49]

[Contrastive Explanation Method][50]

Local

Black-box (*differentiable*), White-box (*TensorFlow*)

Classification

Tabular (numerical), Image

""

[docs][51], [paper][52]

[Counterfactuals Guided by Prototypes][53]

Local

Black-box (*differentiable*), White-box (*TensorFlow*)

Classification

Tabular (numerical, categorical), Image

""

[docs][54], [paper][55]

[Counterfactuals with Reinforcement Learning][56]

Local

Black-box

Classification

Tabular (numerical, categorical), Image

""

[docs][57], [paper][58]

[Similarity explanations][59]

Local

White-box

Classification, Regression

Tabular (numerical, categorical), Text and Image

What are the instances in the training set that are most similar to the instance of interest
according to the model?

[docs][60], [paper][61]

### 1. Global Feature Attribution

Global Feature Attribution methods aim to show the dependency of model output on a subset of the
input features. They provide global insight describing the model's behaviour over the input space.
For instance, Accumulated Local Effects plots obtain graphs that directly visualize the relationship
between feature and prediction over a specific set of samples.

Suppose a trained regression model that predicts the number of bikes rented on a given day depending
on the temperature, humidity, and wind speed. A global feature attribution plot for the temperature
feature might be a line graph plotted against the number of bikes rented. One would anticipate an
increase in rentals until a specific temperature and then a decrease after it gets too hot.

#### Accumulated Local Effects

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Accumulated Local Effects][62]

Global

Black-box

Classification, Regression

Tabular (numerical)

How does model prediction vary with respect to features of interest?

[docs][63], [paper][64]

Alibi provides [accumulated local effects (ALE)][65] plots because they give the most accurate
insight. Alternatives include Partial Dependence Plots (PDP), of which ALE is a natural extension.
Suppose we have a model $f$ and features $X={x_1,... x_n}$. Given a subset of the features $X_S$, we
denote $X_C=X \setminus X_S$. $X_S$ is usually chosen to be of size at most 2 in order to make the
generated plots easy to visualize. PDP works by marginalizing the model's output over the features
we are not interested in, $X_C$. The process of factoring out the $X_C$ set causes the introduction
of artificial data, which can lead to errors. ALE plots solve this by using the conditional
probability distribution instead of the marginal distribution and removing any incorrect output
dependencies due to correlated input variables by accumulating local differences in the model output
instead of averaging them. See the [following][66] for a more expansive explanation.

We illustrate the use of ALE on the [wine-quality dataset][67] which is a tabular numeric dataset
with wine quality as the target variable. Because we want a classification task we split the data
into good and bad classes using 5 as the threshold. We can compute the ALE with Alibi (see
[notebook][68]) by simply using:

Hence, we see the model predicts higher alcohol content wines as being better:

[ALE Plot of wine quality good class probability dependency on alcohols]
ALE Plot for wine quality

**Note 2: Categorical Variables and ALE** while ALE is well-defined on numerical tabular data, it
isn't on categorical data. This is because it's unclear what the difference between two categorical
values should be. If the dataset has a mix of categorical and numerical features, we can always
compute the ALE of the numerical ones.

Pros
Cons

ALE plots are easy to visualize and understand intuitively

Harder to explain the underlying motivation behind the method than PDP plots or M plots

Very general as it is a black-box algorithm

Requires access to the training dataset

Doesn't struggle with dependencies in the underlying features, unlike PD plots

ALE of categorical variables is not well-defined

ALE plots are fast

#### Partial Dependence

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Partial Dependence][69]

Global

Black-box, White-box (scikit-learn)

Classification, Regression

Tabular (numerical, categorical)

How does model prediction vary with respect to features of interest?

[docs][70], [paper][71]

Alibi provides [partial dependence (PD)][72] plots as an alternative to ALE. Following the same
notation as above, we remind the reader that the PD is marginalizing the model's output over the
features we are not interested in, $X_C$. This approach has a direct extension for categorical
features, something that ALE struggle with. Although, the practitioner should be aware of the main
limitation of PD, which is the assumption of feature independence. The process of marginalizing out
the set $X_C$ under the assumption of feature independence might thus include in the computation
predictions for data instances belonging to low probability regions of the features distribution.

Pros
Cons

PD plots are easy to visualize and understand intuitively (easier than ALE)

Struggle with dependencies in the underlying features. In the uncorrelated case the interpretation
might be unclear.

Very general as it is a black-box algorithm

Heterogeneous effects might be hidden (ICE to the rescue)

PD plots are in general fast. Even faster implementation for scikit-learn tree based models

PD plots have causal interpretation. The relationship is causal for the model, but not necessarily
for the real world

Natural extension to categorical features

#### Partial Dependence Variance

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Partial Dependence Variance][73]

Global

Black-box, White-box (scikit-learn)

Classification, Regression

Tabular (numerical, categorical)

What are the most important features globally? How much do features interact globally?

[docs][74], [paper][75]

Alibi provides [partial dependence variance][76] as a way to measure globally the feature importance
and the strength of the feature interactions between pairs of features. Since the method is based on
the partial dependence, the practitioner should be aware that the method inherits its main
limitations (see discussion above).

Pros
Cons

Intuitive motivation for the computation of the feature importance

The feature importance captures only the main effect and ignores possible feature interaction

Very general as it is a black-box algorithm

Can fail to detect feature interaction even though those exist

Fast computation in general. Even faster implementation for scikit-learn tree-based models

Offers standardized procedure to quantify the feature importance (i.e., contrasts with internal
feature importance for some tree-based model)

Offers support for both numerical and categorical features

Can quantify the strength of potential interaction effects

#### Permutation Importance

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Permutation Importance][77]

Global

Black-box

Classification, Regression

Tabular (numerical, categorical)

Which are the most important features globally?

[docs][78], [paper][79]

Alibi provides [permutation importance][80] as a way to measure globally the feature importance. The
computation of the feature importance is based on the degree of model performance degradation when
the feature values within a feature column are permuted. One important behavior that a practitioner
should be aware of is that the importance of correlated features can be split between them.

Pros
Cons

A nice and simple interpretation - the feature importance is the increase/decrease in the model
loss/score when a feature is noise.

Need the ground truth labels

Very general as it is a black-box algorithm

Can be biased towards unrealistic data instances

The feature importance takes into account all the feature interactions

The importance metric is related to the loss/score function

Does not require retraining the model

### 2. Local Necessary Features

Local necessary features tell us what features need to stay the same for a specific instance in
order for the model to give the same classification. In the case of a trained image classification
model, local necessary features for a given instance would be a minimal subset of the image that the
model uses to make its decision. Alibi provides two explainers for computing local necessary
features: [anchors][81] and [pertinent positives][82].

#### Anchors

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Anchors][83]

Local

Black-box

Classification

Tabular (numerical, categorical), Text and Image

Which set of features of a given instance are sufficient to ensure the prediction stays the same?

[docs][84], [paper][85]

Anchors are introduced in [Anchors: High-Precision Model-Agnostic Explanations][86]. More detailed
documentation can be found [here][87].

Let $A$ be a rule (set of predicates) acting on input instances, such that $A(x)$ returns $1$ if all
its feature predicates are true. Consider the [wine quality dataset][88] adjusted by partitioning
the data into good and bad wine based on a quality threshold of 0.5:

[First five rows of wine quality dataset]
First five rows of the dataset
[Illustration of an anchor as a subset of two dimensional feature space.]
Illustration of an anchor

An example of a predicate for this dataset would be a rule of the form: `'alcohol > 11.00'`. Note
that the more predicates we add to an anchor, the fewer instances it applies to, as by doing so, we
filter out more instances of the data. Anchors are sets of predicates associated to a specific
instance $x$ such that $x$ is in the anchor ($A(x)=1$) and any other point in the anchor has the
same classification as $x$ ($z$ such that $A(z) = 1 \implies f(z) = f(x)$ where $f$ is the model).
We're interested in finding the Anchor that contains both the most instances and also $x$.

To construct an anchor using Alibi for tabular data such as the wine quality dataset (see
[notebook][89]), we use:

where `x` is an instance of the dataset classified as good.

**Note**: Alibi also gives an idea of the size (coverage) of the Anchor which is the proportion of
the input space the anchor applies to.

To find anchors Alibi sequentially builds them by generating a set of candidates from an initial
anchor candidate, picking the best candidate of that set and then using that to generate the next
set of candidates and repeating. Candidates are favoured on the basis of the number of instances
they contain that are in the same class as $x$ under $f$. The proportion of instances the anchor
contains that are classified the same as $x$ is known as the *precision* of the anchor. We repeat
the above process until we obtain a candidate anchor with satisfactory precision. If there are
multiple such anchors we choose the one that contains the most instances (as measured by
*coverage*).

To compute which of two anchors is better, Alibi obtains an estimate by sampling from
$\mathcal{D}(z|A)$ where $\mathcal{D}$ is the data distribution. The sampling process is dependent
on the type of data. For tabular data, this process is easy; we can fix the values in the Anchor and
replace the rest with values from points sampled from the dataset.

In the case of textual data, anchors are sets of words that the sentence must include to be **in
the** anchor. To sample from $\mathcal{D}(z|A)$, we need to find realistic sentences that include
those words. To help do this Alibi provides support for three [transformer][90] based language
models: `DistilbertBaseUncased`, `BertBaseUncased`, and `RobertaBase`.

Image data being high-dimensional means we first need to reduce it to a lower dimension. We can do
this using image segmentation algorithms (Alibi supports [felzenszwalb][91] , [slic][92] and
[quickshift][93]) to find super-pixels. The user can also use their own custom defined segmentation
function. We then create the anchors from these super-pixels. To sample from $\mathcal{D}(z|A)$ we
replace those super-pixels that aren't in $A$ with something else. Alibi supports superimposing over
the absent super-pixels with an image sampled from the dataset or taking the average value of the
super-pixel.

The fact that the method requires perturbing and comparing anchors at each stage leads to some
limitations. For instance, the more features, the more candidate anchors you can obtain at each
process stage. The algorithm uses a [beam search][94] among the candidate anchors and solves for the
best $B$ anchors at each stage in the process by framing the problem as a [multi-armed bandit][95].
The runtime complexity is $\mathcal{O}(B \cdot p^2 + p^2 \cdot \mathcal{O}*{MAB[B \cdot p, B]})$
where $p$ is the number of features and $\mathcal{O}* {MAB[B \cdot p, B]}$ is the runtime for the
multi-armed bandit ( see [Molnar][96] for more details).

Similarly, comparing anchors that are close to decision boundaries can require many samples to
obtain a clear winner between the two. Also, note that anchors close to decision boundaries are
likely to have many predicates to ensure the required predictive property. This makes them less
interpretable.

Pros
Cons

Easy to explain as rules are simple to interpret

Time complexity scales as a function of features

Is a black-box method as we need to predict the value of an instance and don't need to access model
internals

Requires a large number of samples to distinguish anchors close to decision boundaries

The coverage of an anchor gives a level of global insight as well

Anchors close to decision boundaries are less likely to be interpretable

High dimensional feature spaces such as images need to be reduced to improve the runtime complexity

Practitioners may need domain-specific knowledge to correctly sample from the conditional
probability

#### Contrastive Explanation Method (Pertinent Positives)

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Pertinent Positives][97]

Local

Black-box, White-box (*TensorFlow*)

Classification

Tabular (numerical), Image

Which set of features of a given instance is sufficient to ensure the prediction stays the same

[docs][98], [paper][99]

Introduced by [Amit Dhurandhar, et al][100], a Pertinent Positive is the subset of features of an
instance that still obtains the same classification as that instance. These differ from
[anchors][101] primarily in the fact that they aren't constructed to maximize coverage. The method
to create them is also substantially different. The rough idea is to define an **absence of a
feature** and then perturb the instance to take away as much information as possible while still
retaining the original classification. Note that these are a subset of the [CEM][102] method which
is also used to construct [pertinent negatives/counterfactuals][103].

[Pertinent postive of an MNIST digit]

Given an instance $x$ we use gradient descent to find a $\delta$ that minimizes the following loss:

$AE$ is an [autoencoder][104] generated from the training data. If $\delta$ strays from the original
data distribution, the autoencoder loss will increase as it will no longer reconstruct $\delta$
well. Thus, we ensure that $\delta$ remains close to the original dataset distribution.

Note that $\delta$ is constrained to only "take away" features from the instance $x$. There is a
slightly subtle point here: removing features from an instance requires correctly defining
non-informative feature values. For the [MNIST digits][105], it's reasonable to assume that the
black background behind each digit represents an absence of information. In general, having to
choose a non-informative value for each feature is non-trivial and domain knowledge is required.
This is the reverse to the [contrastive explanation method (pertinent-negatives)][106] method
introduced in the section on [counterfactual instances][107].

Note that we need to compute the loss gradient through the model. If we have access to the
internals, we can do this directly. Otherwise, we need to use numerical differentiation at a high
computational cost due to the extra model calls we need to make. This does however mean we can use
this method for a wide range of black-box models but not all. We require the model to be
differentiable which isn't always true. For instance tree-based models have piece-wise constant
output.

Pros
Cons

Can be used with both white-box (TensorFlow) and some black-box models

Finding non-informative feature values to take away from an instance is often not trivial, and
domain knowledge is essential

The autoencoder loss requires access to the original dataset

Need to tune hyperparameters $\beta$ and $\gamma$

The insight doesn't tell us anything about the coverage of the pertinent positive

Slow for black-box models due to having to numerically evaluate gradients

Only works for differentiable black-box models

### 3. Local Feature Attribution

Local feature attribution (LFA) asks how each feature in a given instance contributes to its
prediction. In the case of an image, this would highlight those pixels that are most responsible for
the model prediction. Note that this differs subtly from [Local Necessary Features][108] which find
the *minimal subset* of features required to keep the same prediction. Local feature attribution
instead assigns a score to each feature.

A good example use of local feature attribution is to detect that an image classifier is focusing on
the correct features of an image to infer the class. In their paper ["Why Should I Trust You?":
Explaining the Predictions of Any Classifier][109], Marco Tulio Ribeiro et al. train a logistic
regression classifier on a small dataset of images of wolves and huskies. The data set has been
handpicked so that only the pictures of wolves have snowy backdrops while the huskies don't. LFA
methods reveal that the resulting misclassification of huskies in snow as wolves results from the
network incorrectly focusing on those images snowy backdrops.

[Husky with snowy backdrop misclassified as wolf]

*Figure 11 from "Why Should I Trust You?": Explaining the Predictions of Any Classifier.*

This gives:

[IG applied to Wine quality dataset for class Good]

{% hint style="info" %} **Note 4: Comparison to ALE**

The alcohol feature value contributes negatively here to the "Good" prediction which seems to
contradict the [ALE result][110]. However, The instance $x$ we choose has an alcohol content of
9.4%, which is reasonably low for a wine classed as "Good" and is consistent with the ALE plot. (The
median for good wines is 10.8% and bad wines 9.7%) {% endhint %}

Pros
Cons

Simple to understand and visualize, especially with image data

White-box method. Requires the partial derivatives of the model outputs with respect to inputs

Doesn't require access to the training data

Requires [choosing the baseline][111] which can have a significant effect on the outcome

[Satisfies several desirable properties][112]

#### Kernel SHAP

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Kernel SHAP][113]

Local

Black-box

Classification, Regression

Tabular (numerical, categorical)

What does each feature contribute to the model prediction?

[docs][114], [paper][115]

[Kernel SHAP][116] ([Alibi method docs][117]) is a method for computing the Shapley values of a
model around an instance. [Shapley values][118] are a game-theoretic method of assigning payout to
players depending on their contribution to an overall goal. In our case, the features are the
players, and the payouts are the attributions.

Given any subset of features, we can ask how a feature's presence in that set contributes to the
model output. We do this by computing the model output for the set with and without the specific
feature. We obtain the Shapley value for that feature by considering these contributions with and
without it present for all possible subsets of features.

Two problems arise. Most models are not trained to take a variable number of input features. And
secondly, considering all possible sets of absent features leads to considering the [power set][119]
which is prohibitively large when there are many features.

To solve the former, we sample from the **interventional conditional expectation**. This replaces
missing features with values sampled from the training distribution. And to solve the latter, the
kernel SHAP method samples on the space of subsets to obtain an estimate.

A downside of interfering in the distribution like this is that doing so introduces unrealistic
samples if there are dependencies between the features.

Alibi provides a wrapper to the [SHAP library][120]. We can use this explainer to compute the
Shapley values for a [sklearn][121] [random forest][122] model using the following (see
[notebook][123]):

This gives the following output:

[Kernel SHAP applied to Wine quality dataset for class Good]

This result is similar to the one for [Integrated Gradients][124] although there are differences due
to using different methods and models in each case.

Pros
Cons

[Satisfies several desirable properties][125]

Kernel SHAP is slow owing to the number of samples required to estimate the Shapley values
accurately

Shapley values can be easily interpreted and visualized

The interventional conditional probability introduces unrealistic data points

Very general as is a black-box method

Requires access to the training dataset

#### Path-dependent Tree SHAP

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Tree SHAP (path-dependent)][126]

Local

White-box (*XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models*)

Classification, Regression

Tabular (numerical, categorical)

What does each feature contribute to the model prediction?

[docs][127], [paper][128]

Computing the Shapley values for a model requires computing the interventional conditional
expectation for each member of the [power set][129] of instance features. For tree-based models we
can approximate this distribution by applying the tree as usual. However, for missing features, we
take both routes down the tree, weighting each path taken by the proportion of samples from the
training dataset that go each way. The tree SHAP method does this simultaneously for all members of
the feature power set, obtaining a [significant speedup][130] . Assume the random forest has $T$
trees, with a depth of $D$, let $L$ be the number of leaves and let $M$ be the size of the feature
set. If we compute the approximation for each member of the power set we obtain a time complexity of
$O( TL2^M)$. In contrast, computing for all sets simultaneously we achieve $O(TLD^2)$.

To compute the path-dependent tree SHAP explainer for a random forest using Alibi ( see
[notebook][131]) we use:

From this we obtain:

[Path-dependent tree SHAP applied to Wine quality dataset for class Good]

This result is similar to the one for [Integrated Gradients][132] and [Kernel SHAP][133] although
there are differences due to using different methods and models in each case.

Pros
Cons

[Satisfies several desirable properties][134]

Only applies to tree-based models

Very fast for a valuable category of models

Uses an approximation of the interventional conditional expectation instead of computing it directly

Doesn't require access to the training data

Shapley values can be easily interpreted and visualized

#### Interventional Tree SHAP

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Tree SHAP (interventional)][135]

Local

White-box (*XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models*)

Classification, Regression

Tabular (numerical, categorical)

What does each feature contribute to the model prediction?

[docs][136], [paper][137]

Suppose we sample a reference data point, $r$, from the training dataset. Let $F$ be the set of all
features. For each feature, $i$, we then enumerate over all subsets of $S\subset F \setminus {i}$.
If a subset is missing a feature, we replace it with the corresponding one in the reference sample.
We can then compute $f(S)$ directly for each member of the power set of instance features to get the
Shapley values.

Enforcing independence of the $S$ and $F\setminus S$ in this way is known as intervening in the
underlying data distribution and is the source of the algorithm's name. Note that this breaks any
independence between features in the dataset, which means the data points we're sampling won't
always be realistic.

For a single tree and sample $r$ if we iterate over all the subsets of $S \subset F \setminus {i}$,
it would give $O( M2^M)$ time complexity. The interventional tree SHAP algorithm runs with
[$O(TLD)$][138] time complexity.

The main difference between the interventional and the path-dependent tree SHAP methods is that the
latter approximates the interventional conditional expectation, whereas the former method calculates
it directly.

To compute the interventional tree SHAP explainer for a random forest using Alibi ( see
[notebook][139]), we use:

From this we obtain:

[Interventional tree SHAP applied to Wine quality dataset for class Good]

This result is similar to the one for [Integrated Gradients][140], [Kernel SHAP][141] ,
[Path-dependent Tree SHAP][142] although there are differences due to using different methods and
models in each case.

For a great interactive explanation of the interventional Tree SHAP method [see][143].

Pros
Cons

[Satisfies several desirable properties][144]

Only applies to tree-based models

Very fast for a valuable category of models

Requires access to the dataset

Shapley values can be easily interpreted and visualized

Typically slower than the path-dependent method

Computes the interventional conditional expectation exactly unlike the path-dependent method

### 4. Counterfactual instances

Given an instance of the dataset and a prediction given by a model, a question naturally arises how
would the instance minimally have to change for a different prediction to be provided. Such a
generated instance is known as a *counterfactual*. Counterfactuals are local explanations as they
relate to a single instance and model prediction.

Given a classification model trained on the MNIST dataset and a sample from the dataset, a
counterfactual would be a generated image that closely resembles the original but is changed enough
that the model classifies it as a different number from the original instance.

[Samples from MNIST and counterfactuals for each]

*From Samoilescu RF et al., Model-agnostic and Scalable Counterfactual Explanations via
Reinforcement Learning, 2021*

Counterfactuals can be used to both [debug and augment][145] model functionality. Given tabular data
that a model uses to make financial decisions about a customer, a counterfactual would explain how
to change their behavior to obtain a different conclusion. Alternatively, it may tell the Machine
Learning Engineer that the model is drawing incorrect assumptions if the recommended changes involve
features that are irrelevant to the given decision. However, practitioners must still be wary of
[bias][146].

A counterfactual, $x_{\text{cf}}$, needs to satisfy

* The model prediction on $x_{\text{cf}}$ needs to be close to the pre-defined output (e.g. desired
  class label).
* The counterfactual $x_{\text{cf}}$ should be interpretable.

The first requirement is clear. The second, however, requires some idea of what interpretable means.
Alibi exposes four methods for finding counterfactuals: [**counterfactual instances (CFI)**][147] ,
[**contrastive explanations (CEM)**][148] , [**counterfactuals guided by prototypes (CFP)**][149],
and [**counterfactuals with reinforcement learning (CFRL)**][150]. Each of these methods deals with
interpretability slightly differently. However, all of them require sparsity of the solution. This
means we prefer to only change a small subset of the features which limits the complexity of the
solution making it more understandable.

Note that sparse changes to the instance of interest doesn't guarantee that the generated
counterfactual is believably a member of the data distribution. [**CEM**][151] , [**CFP**][152], and
[**CFRL**][153] also require that the counterfactual be in distribution in order to be
interpretable.

[Examples of counterfactuals constructed using CFI and CFP methods]

*Original MNIST 7 instance, Counterfactual instances constructed using 1. ****counterfactual
instances**** method, 2.****counterfactual instances with prototypes**** method*

The first three methods [**CFI**][154] , [**CEM**][155] , [**CFP**][156] all construct
counterfactuals using a very similar method. They build them by defining a loss that prefer
interpretable instances close to the target class. They then use gradient descent to move within the
feature space until they obtain a counterfactual of sufficient quality. The main difference is the
**CEM** and **CFP** methods also train an autoencoder to ensure that the constructed counterfactuals
are within the data-distribution.

[Construction of different types of interpretable counterfactuals]

*Obtaining counterfactuals using gradient descent with and without autoencoder trained on data
distribution*

These three methods only realistically work for grayscale images and anything multi-channel will not
be interpretable. In order to get quality results for multi-channel images practitioners should use
[CFRL][157].

[CFRL][158] uses a similar loss to CEM and CFP but applies reinforcement learning to train a model
which will generate counterfactuals on demand.

{% hint style="info" %} **Note 5: fit and explain method runtime differences** Alibi explainers
expose two methods, `fit` and `explain`. Typically in machine learning the method that takes the
most time is the fit method, as that's where the model optimization conventionally takes place. In
explainability, the explain step often requires the bulk of computation. However, this isn't always
the case.

Among the explainers in this section, there are two approaches taken. The first finds a
counterfactual when the user requests the insight. This happens during the `.explain()` method call
on the explainer class. This is done by running gradient descent on model inputs to find a
counterfactual. The methods that take this approach are **counterfactual instances**, **contrastive
explanation**, and **counterfactuals guided by prototypes**. Thus, the `fit` method in these cases
is quick, but the `explain` method is slow.

The other approach, **counterfactuals with reinforcement learning**, trains a model that produces
explanations on demand. The training takes place during the `fit` method call, so this has a long
runtime while the `explain` method is quick. If you want performant explanations in production
environments, then the latter approach is preferable. {% endhint %}

#### Counterfactual Instances

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Counterfactual Instances][159]

Local

Black-box (*differentiable*), White-box (*TensorFlow*)

Classification

Tabular(numerical), Image

What minimal change to features is required to reclassify the current prediction?

[docs][160], [paper][161]

Let the model be given by $f$, and let $p_t$ be the target probability of class $t$. Let $\lambda$
be a hyperparameter. This method constructs counterfactual instances from an instance $X$ by running
gradient descent on a new instance $X'$ to minimize the following loss:

L(X′,X)=(ft(X′)−pt)2+λL1(X′,X)L(X', X)= (f_{t}(X') - p_{t})^2 + \lambda L_{1}(X',
X)L(X′,X)=(ft​(X′)−pt​)2+λL1​(X′,X)

The first term pushes the constructed counterfactual towards the desired class, and the use of the
$L_{1}$ norm encourages sparse solutions.

This method requires computing gradients of the loss in the model inputs. If we have access to the
model and the gradients are available, this can be done directly. If not, we can use numerical
gradients, although this comes at a considerable performance cost.

A problem arises here in that encouraging sparse solutions doesn't necessarily generate
interpretable counterfactuals. This happens because the loss doesn't prevent the counterfactual
solution from moving off the data distribution. Thus, you will likely get an answer that doesn't
look like something that you would expect to see from the data.

To use the counterfactual instances method from Alibi applied to the wine quality dataset (see
[notebook][162]), use:

Gives the expected result:

Pros
Cons

Both a black-box and white-box method

Not likely to give human interpretable instances

Doesn't require access to the training dataset

Requires tuning of $\lambda$ hyperparameter

Slow for black-box models due to having to numerically evaluate gradients

(contrastive-explanation-method-pertinent-negatives)=

#### Contrastive Explanation Method (Pertinent Negatives)

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Contrastive Explanation Method][163]

Local

Black-box (*differentiable*), White-box (*TensorFlow*)

Classification

Tabular(numerical), Image

What minimal change to features is required to reclassify the current prediction?

[docs][164], [paper][165]

CEM follows a similar approach to the above but includes three new details. Firstly an elastic net
$\beta L_{1} + L_{2}$ regularizer term is added to the loss. This term causes the solutions to be
both close to the original instance and sparse.

Secondly, we require that $\delta$ only adds new features rather than takes them away. We need to
define what it means for a feature to be present so that the perturbation only works to add and not
remove them. In the case of the MNIST dataset, an obvious choice of "present" feature is if the
pixel is equal to 1 and absent if it is equal to 0. This is simple in the case of the MNIST data set
but more difficult in complex domains such as colour images.

Thirdly, by training an autoencoder to penalize counterfactual instances that deviate from the data
distribution. This works by minimizing the reconstruction loss of the autoencoder applied to
instances. If a generated instance is unlike anything in the dataset, the autoencoder will struggle
to recreate it well, and its loss term will be high. We require three hyperparameters $c$, $\beta$
and $\gamma$ to define the following loss:

A subtle aspect of this method is that it requires defining the absence or presence of features as
delta is restrained only to allow you to add information. For the MNIST digits, it's reasonable to
assume that the black background behind each written number represents an absence of information.
Similarly, in the case of colour images, you might take the median pixel value to convey no
information, and moving away from this value adds information. For numerical tabular data, we can
use the feature mean. In general, choosing a non-informative value for each feature is non-trivial,
and domain knowledge is required. This is the reverse process to the [contrastive explanation method
(pertinent-positives)][166] method introduced in the section on [local necessary features][167] in
which we take away features rather than add them.

This approach extends the definition of interpretable to include a requirement that the computed
counterfactual be believably a member of the dataset. This isn't always satisfied (see image below).
In particular, the constructed counterfactual often doesn't look like a member of the target class.

[Example of less interpretable result obtained by CEM]

*An original MNIST instance and a pertinent negative obtained using CEM.*

To compute a pertinent-negative using Alibi (see [notebook][168]) we use:

Gives the expected result:

This method can apply to both black-box and white-box models. There is a performance cost from
computing the numerical gradients in the black-box case due to having to numerically evaluate
gradients.

Pros
Cons

Provides more interpretable instances than the counterfactual instances' method.

Requires access to the dataset to train the autoencoder

Applies to both white and black-box models

Requires setup and configuration in choosing $c$, $\gamma$ and $\beta$

Requires training an autoencoder

Requires domain knowledge when choosing what it means for a feature to be present or not

Slow for black-box models

#### Counterfactuals Guided by Prototypes

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Counterfactuals Guided by Prototypes][169]

Local

Black-box (*differentiable*), White-box (*TensorFlow*)

Classification

Tabular (numerical, categorical), Image

What minimal change to features is required to reclassify the current prediction?

[docs][170], [paper][171]

For this method, we add another term to the loss that optimizes for the distance between the
counterfactual instance and representative members of the target class. In doing this, we require
interpretability also to mean that the generated counterfactual is believably a member of the target
class and not just in the data distribution.

With hyperparameters $c$, $\gamma$ and $\beta$, the loss is given by:

proto){proto})proto)

This method produces much more interpretable results than [CFI][172] and [CEM][173].

Because the prototype term steers the solution, we can remove the prediction loss term. This makes
this method much faster if we are using a black-box model as we don't need to compute the gradients
numerically. However, occasionally the prototype isn't a member of the target class. In this case
you'll end up with an incorrect counterfactual.

To use the counterfactual with prototypes method in Alibi (see [notebook][174]) we do:

We get the following results:

Pros
Cons

Generates more interpretable instances than the CEM method

Requires access to the dataset

Black-box version of the method is fast

Requires setup and configuration in choosing $\gamma$, $\beta$ and $c$

Applies to more data-types

Requires training an autoencoder

#### Counterfactuals with Reinforcement Learning

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Counterfactuals with Reinforcement Learning][175]

Local

Black-box

Classification

Tabular (numerical, categorical), Image

What minimal change to features is required to reclassify the current prediction?

[docs][176], [paper][177]

This black-box method splits from the approach taken by the above three significantly. Instead of
minimizing a loss during the explain method call, it trains a **new model** when **fitting** the
explainer called an **actor** that takes instances and produces counterfactuals. It does this using
**reinforcement learning**. In reinforcement learning, an actor model takes some state as input and
generates actions; in our case, the actor takes an instance with a target classification and
attempts to produce a member of the target class. Outcomes of actions are assigned rewards dependent
on a reward function designed to encourage specific behaviors. In our case, we reward correctly
classified counterfactuals generated by the actor. As well as this, we reward counterfactuals that
are close to the data distribution as modeled by an autoencoder. Finally, we require that they are
sparse perturbations of the original instance. The reinforcement training step pushes the actor to
take high reward actions. CFRL is a black-box method as the process by which we update the actor to
maximize the reward only requires estimating the reward via sampling the counterfactuals.

As well as this, CFRL actors can be trained to ensure that certain **constraints** can be taken into
account when generating counterfactuals. This is highly desirable as a use case for counterfactuals
is to suggest the necessary changes to an instance to obtain a different classification. In some
cases, you want these changes to be constrained, for instance, when dealing with immutable
characteristics. In other words, if you are using the counterfactual to advise changes in behavior,
you want to ensure the changes are enactable. Suggesting that someone needs to be two years younger
to apply for a loan isn't very helpful.

The training process requires randomly sampling data instances, along with constraints and target
classifications. We can then compute the reward and update the actor to maximize it. We do this
without needing access to the model internals; we only need to obtain a prediction in each case. The
end product is a model that can generate interpretable counterfactual instances at runtime with
arbitrary constraints.

To use CFRL on the wine dataset (see [notebook][178]), we use:

Which gives the following output:

{% hint style="info" %} **Note 6: CFRL explainers** Alibi exposes two explainer methods for
counterfactuals with reinforcement learning. The first is the CounterfactualRL and the second is
CounterfactualRlTabular. The difference is that CounterfactualRlTabular is designed to support
categorical features. See the [CFRL documentation page][179] for more details. {% endhint %}

Pros
Cons

Generates more interpretable instances than the CEM method

Longer to fit the model

Very fast at runtime

Requires to fit an autoencoder

Can be trained to account for arbitrary constraints

Requires access to the training dataset

General as is a black-box algorithm

#### Counterfactual Example Results

For each of the four explainers, we have generated a counterfactual instance. We compare the
original instance to each:

Feature
Instance
CFI
CEM
CFP
CFRL

sulphates

0.67

**0.64**

**0.549**

**0.623**

**0.598**

alcohol

10.5

**9.88**

**9.652**

**9.942**

**9.829**

residual sugar

1.6

**1.582**

**1.479**

1.6

**2.194**

chlorides

0.062

0.061

**0.057**

0.062

**0.059**

free sulfur dioxide

5.0

**4.955**

**2.707**

5.0

**6.331**

total sulfur dioxide

12.0

**11.324**

12.0

12.0

**14.989**

fixed acidity

9.2

**9.23**

9.2

9.2

**8.965**

volatile acidity

0.36

0.36

0.36

0.36

**0.349**

citric acid

0.34

0.334

0.34

0.34

0.242

density

0.997

0.997

0.997

0.997

0.997

pH

3.2

3.199

3.2

3.2

3.188

The CFI, CEM, and CFRL methods all perturb more features than CFP, making them less interpretable.
Looking at the ALE plots, we can see how the counterfactual methods change the features to flip the
prediction. In general, each method seems to decrease the sulphates and alcohol content to obtain a
"bad" classification consistent with the ALE plots. Note that the ALE plots potentially miss details
local to individual instances as they are global insights.

[Ale plots for those features that the above counterfactuals have changed the most.]

### 5. Similarity explanations

Explainer
Scope
Model types
Task types
Data types
Use
Resources

[Similarity explanations][180]

Local

White-box

Classification, Regression

Tabular (numerical, categorical), Text and Image

What are the instances in the training set that are most similar to the instance of interest
according to the model?

[docs][181], [paper][182]

Similarity explanations are instance-based explanations that focus on training data points to
justify a model prediction on a test instance. Given a trained model and a test instance whose
prediction is to be explained, these methods scan the training set, finding the most similar data
points according to the model which forms an explanation. This type of explanation can be
interpreted as the model justifying its prediction by referring to similar instances which may share
the same prediction---*"I classify this image as a 'Golden Retriever' because it is most similar to
images in the training set which I also classified as 'Golden Retriever'"*.

[A similarity explanation justifies the classification of an image as a 'Golden Retriever' because
most similar instances in the training set are also classified as 'Golden Retriever'.]

*A similarity explanation justifies the classification of an image as a 'Golden Retriever' because
most similar instances in the training set are also classified as 'Golden Retriever'.*

[NextGetting Started][183]

Last updated 2 months ago

Was this helpful?

[1]: https://github.com/SeldonIO/alibi/blob/master/docs-gb/source/overview/high_level.md
[2]: /alibi-explain/overview
[3]: https://en.wikipedia.org/wiki/Regression_analysis
[4]: https://en.wikipedia.org/wiki/Statistical_classification
[5]: https://en.wikipedia.org/wiki/Neural_network
[6]: https://en.wikipedia.org/wiki/Random_forest
[7]: https://www.mckinsey.com/business-functions/mckinsey-analytics/our-insights/global-survey-the-s
tate-of-ai-in-2020
[8]: https://onlinelibrary.wiley.com/doi/abs/10.1002/bdm.542
[9]: https://arxiv.org/abs/1909.06342
[10]: http://proceedings.mlr.press/v70/sundararajan17a.html
[11]: /alibi-explain#biases
[12]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/path-depende
nt-tree-shap/README.md
[13]: https://en.wikipedia.org/wiki/Decision_tree_learning
[14]: /alibi-explain/overview/white_box_black_box
[15]: https://dl.acm.org/doi/abs/10.1145/3351095.3375624
[16]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/anchors/READ
ME.md
[17]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/accumulated-
local-effects/README.md
[18]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/ALE.ipynb
[19]: https://arxiv.org/abs/1612.08468
[20]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/partial-depe
ndence/README.md
[21]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PartialDepend
ence.ipynb
[22]: https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-appr
oximation-A-gradient-boostingmachine/10.1214/aos/1013203451.full
[23]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/partial-depe
ndence-variance/README.md
[24]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PartialDepend
enceVariance.ipynb
[25]: https://arxiv.org/abs/1805.04755
[26]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/permutation-
importance/README.md
[27]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PermutationIm
portance.ipynb
[28]: https://arxiv.org/abs/1801.01489
[29]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/anchors/READ
ME.md
[30]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/Anchors.ipynb
[31]: https://dl.acm.org/doi/abs/10.5555/3504035.3504222
[32]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/contrastive-
explanation-method-pertinent-positives/README.md
[33]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CEM.ipynb
[34]: https://arxiv.org/abs/1802.07623
[35]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/integrated-g
radients/README.md
[36]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/IntegratedGra
dients.ipynb
[37]: https://arxiv.org/abs/1703.01365
[38]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/kernel-shap/
README.md
[39]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/KernelSHAP.ip
ynb
[40]: https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
[41]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/path-depende
nt-tree-shap/README.md
[42]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/TreeSHAP.ipyn
b
[43]: https://www.nature.com/articles/s42256-019-0138-9
[44]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/intervention
al-tree-shap/README.md
[45]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/TreeSHAP.ipyn
b
[46]: https://www.nature.com/articles/s42256-019-0138-9
[47]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfactu
al-instances/README.md
[48]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CF.ipynb
[49]: https://arxiv.org/abs/1711.00399
[50]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/contrastive-
explanation-method-pertinent-negatives/README.md
[51]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CEM.ipynb
[52]: https://arxiv.org/abs/1802.07623
[53]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfactu
als-guided-by-prototypes/README.md
[54]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFProto.ipynb
[55]: https://arxiv.org/abs/1907.02584
[56]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfactu
als-with-reinforcement-learning/README.md
[57]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFRL.ipynb
[58]: https://arxiv.org/abs/2106.02597
[59]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/similarity-e
xplanations/README.md
[60]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/Similarity.ip
ynb
[61]: https://papers.nips.cc/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html
[62]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/ALE.ipynb
[63]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/ALE.ipynb
[64]: https://arxiv.org/abs/1612.08468
[65]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/ALE.ipynb
[66]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/ALE.ipynb
[67]: https://archive.ics.uci.edu/ml/datasets/wine+quality
[68]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/overview.ipy
nb
[69]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PartialDepend
ence.ipynb
[70]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PartialDepend
ence.ipynb
[71]: https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-appr
oximation-A-gradient-boostingmachine/10.1214/aos/1013203451.full
[72]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PartialDepend
ence.ipynb
[73]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/partial-depe
ndence-variance/README.md
[74]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PartialDepend
enceVariance.ipynb
[75]: https://arxiv.org/abs/1805.04755
[76]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PartialDepend
enceVariance.ipynb
[77]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/permutation-
importance/README.md
[78]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PermutationIm
portance.ipynb
[79]: https://arxiv.org/abs/1801.01489
[80]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/PermutationIm
portance.ipynb
[81]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/anchors/READ
ME.md
[82]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/contrastive-
explanation-method-pertinent-positives/README.md
[83]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/Anchors.ipynb
[84]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/Anchors.ipynb
[85]: https://dl.acm.org/doi/abs/10.5555/3504035.3504222
[86]: https://dl.acm.org/doi/abs/10.5555/3504035.3504222
[87]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/Anchors.ipynb
[88]: https://archive.ics.uci.edu/ml/datasets/wine+quality
[89]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/overview.ipy
nb
[90]: https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)
[91]: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#felzenszw
alb-s-efficient-graph-based-segmentation
[92]: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#slic-k-me
ans-based-image-segmentation
[93]: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#quickshif
t-image-segmentation
[94]: https://en.wikipedia.org/wiki/Beam_search
[95]: https://en.wikipedia.org/wiki/Multi-armed_bandit
[96]: https://christophm.github.io/interpretable-ml-book/anchors.html#complexity-and-runtime
[97]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CEM.ipynb
[98]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CEM.ipynb
[99]: https://arxiv.org/abs/1802.07623
[100]: https://arxiv.org/abs/1802.07623
[101]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/anchors/REA
DME.md
[102]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CEM.ipynb
[103]: /alibi-explain#4-counterfactual-instances
[104]: https://en.wikipedia.org/wiki/Autoencoder
[105]: http://yann.lecun.com/exdb/mnist/
[106]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/contrastive
-explanation-method-pertinent-negatives/README.md
[107]: /alibi-explain#4-counterfactual-instances
[108]: /alibi-explain#2-local-necessary-features
[109]: https://arxiv.org/abs/1602.04938
[110]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/ale-plot/RE
ADME.md
[111]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/choice-of-b
aseline/README.md
[112]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/lfa-propert
ies/README.md
[113]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/KernelSHAP.i
pynb
[114]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/KernelSHAP.i
pynb
[115]: https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
[116]: https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
[117]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/KernelSHAP.i
pynb
[118]: https://christophm.github.io/interpretable-ml-book/shapley.html
[119]: https://en.wikipedia.org/wiki/Power_set
[120]: https://github.com/slundberg/shap
[121]: https://scikit-learn.org/stable/
[122]: https://en.wikipedia.org/wiki/Random_forest
[123]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/overview.ip
ynb
[124]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/comparison-
to-ale/README.md
[125]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/lfa-propert
ies/README.md
[126]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/TreeSHAP.ipy
nb
[127]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/TreeSHAP.ipy
nb
[128]: https://www.nature.com/articles/s42256-019-0138-9
[129]: https://en.wikipedia.org/wiki/Power_set
[130]: https://www.researchgate.net/publication/333077391_Explainable_AI_for_Trees_From_Local_Explan
ations_to_Global_Understanding
[131]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/overview.ip
ynb
[132]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/comparison-
to-ale/README.md
[133]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/kern-shap-p
lot/README.md
[134]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/lfa-propert
ies/README.md
[135]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/TreeSHAP.ipy
nb
[136]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/TreeSHAP.ipy
nb
[137]: https://www.nature.com/articles/s42256-019-0138-9
[138]: https://hughchen.github.io/its_blog/index.html
[139]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/overview.ip
ynb
[140]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/comparison-
to-ale/README.md
[141]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/kern-shap-p
lot/README.md
[142]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/pd-tree-sha
p-plot/README.md
[143]: https://hughchen.github.io/its_blog/index.html
[144]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/lfa-propert
ies/README.md
[145]: https://research-information.bris.ac.uk/en/publications/counterfactual-explanations-of-machin
e-learning-predictions-oppor
[146]: /alibi-explain#biases
[147]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
ual-instances/README.md
[148]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/contrastive
-explanation-method-pertinent-negatives/README.md
[149]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
uals-guided-by-prototypes/README.md
[150]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
uals-with-reinforcement-learning/README.md
[151]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/contrastive
-explanation-method-pertinent-negatives/README.md
[152]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
uals-guided-by-prototypes/README.md
[153]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
uals-with-reinforcement-learning/README.md
[154]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
ual-instances/README.md
[155]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/contrastive
-explanation-method-pertinent-negatives/README.md
[156]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
uals-guided-by-prototypes/README.md
[157]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
uals-with-reinforcement-learning/README.md
[158]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
uals-with-reinforcement-learning/README.md
[159]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CF.ipynb
[160]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CF.ipynb
[161]: https://arxiv.org/abs/1711.00399
[162]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/overview.ip
ynb
[163]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CEM.ipynb
[164]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CEM.ipynb
[165]: https://arxiv.org/abs/1802.07623
[166]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/contrastive
-explanation-method-pertinent-positives/README.md
[167]: /alibi-explain#2-local-necessary-features
[168]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/overview.ip
ynb
[169]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFProto.ipyn
b
[170]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFProto.ipyn
b
[171]: https://arxiv.org/abs/1907.02584
[172]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/counterfact
ual-instances/README.md
[173]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/contrastive
-explanation-method-pertinent-negatives/README.md
[174]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/overview.ip
ynb
[175]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFRL.ipynb
[176]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFRL.ipynb
[177]: https://arxiv.org/abs/2106.02597
[178]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/overview.ip
ynb
[179]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/CFRL.ipynb
[180]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/overview/similarity-
explanations/README.md
[181]: https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/methods/Similarity.i
pynb
[182]: https://papers.nips.cc/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html
[183]: /alibi-explain/overview/getting_started
