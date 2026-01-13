In [1]:
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from __future__ import print_function

## Fetching data, training a classifier[¶][1]

For this tutorial, we'll be using the [20 newsgroups dataset][2]. In particular, for simplicity,
we'll use a 2-class subset: atheism and christianity.

In [2]:
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

Let's use the tfidf vectorizer, commonly used for text.

In [3]:
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

Now, let's say we want to use random forests for classification. It's usually hard to understand
what random forests are doing, especially with many trees.

In [4]:
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)
Out[4]:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
In [5]:
pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')
Out[5]:
0.92093023255813955

We see that this classifier achieves a very high F score. [The sklearn guide to 20 newsgroups][3]
indicates that Multinomial Naive Bayes overfits this dataset by learning irrelevant stuff, such as
headers. Let's see if random forests do the same.

## Explaining predictions using lime[¶][4]

Lime explainers assume that classifiers act on raw text, but sklearn classifiers act on vectorized
representation of texts. For this purpose, we use sklearn's pipeline, and implements predict_proba
on raw_text lists.

In [6]:
from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)
In [7]:
print(c.predict_proba([newsgroups_test.data[0]]))
[[ 0.274  0.726]]

Now we create an explainer object. We pass the class_names a an argument for prettier display.

In [8]:
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)

We then generate an explanation with at most 6 features for an arbitrary document in the test set.

In [9]:
idx = 83
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])
Document id: 83
Probability(christian) = 0.414
True class: atheism

The classifier got this example right (it predicted atheism).
The explanation is presented below as a list of weighted features.

In [10]:
exp.as_list()
Out[10]:
[(u'Posting', -0.15748303818990594),
 (u'Host', -0.13220892468795911),
 (u'NNTP', -0.097422972255878093),
 (u'edu', -0.051080418945152584),
 (u'have', -0.010616558305370854),
 (u'There', -0.0099743822272458232)]

These weighted features are a linear model, which approximates the behaviour of the random forest
classifier in the vicinity of the test example. Roughly, if we remove 'Posting' and 'Host' from the
document , the prediction should move towards the opposite class (Christianity) by about 0.27 (the
sum of the weights for both features). Let's see if this is the case.

In [11]:
print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])
Original prediction: 0.414
Prediction removing some features: 0.684
Difference: 0.27

Pretty close!
The words that explain the model around this document seem very arbitrary - not much to do with
either Christianity or Atheism.
In fact, these are words that appear in the email headers (you will see this clearly soon), which
make distinguishing between the classes much easier.

## Visualizing explanations[¶][5]

The explanations can be returned as a matplotlib barplot:

In [12]:
%matplotlib inline
fig = exp.as_pyplot_figure()

The explanations can also be exported as an html page (which we can render here in this notebook),
using D3.js to render graphs.

In [13]:
exp.show_in_notebook(text=False)

Alternatively, we can save the fully contained html page to a file:

In [14]:
exp.save_to_file('/tmp/oi.html')

Finally, we can also include a visualization of the original document, with the words in the
explanations highlighted. Notice how the words that affect the classifier the most are all in the
email header.

In [15]:
exp.show_in_notebook(text=True)

That's it for this tutorial. Random forests were just an example, this explainer works for any
classifier you may want to use, as long as it implements predict_proba.

[1]: #Fetching-data,-training-a-classifier
[2]: http://scikit-learn.org/stable/datasets/#the-20-newsgroups-text-dataset
[3]: http://scikit-learn.org/stable/datasets/#filtering-text-for-more-realistic-training
[4]: #Explaining-predictions-using-lime
[5]: #Visualizing-explanations
