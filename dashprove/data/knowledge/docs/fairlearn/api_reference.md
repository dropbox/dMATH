# API Docs[#][1]

# [`fairlearn.datasets`][2][#][3]

This module contains datasets that can be used for benchmarking and education.

──────────────────────────┬─────────────────────────────────────────────────────────────────────────
[`fetch_acs_income`][4]   │Load the ACS Income dataset (regression).                                
──────────────────────────┼─────────────────────────────────────────────────────────────────────────
[`fetch_adult`][5]        │Load the UCI Adult dataset (binary classification).                      
──────────────────────────┼─────────────────────────────────────────────────────────────────────────
[`fetch_bank_marketing`][6│Load the UCI bank marketing dataset (binary classification).             
]                         │                                                                         
──────────────────────────┼─────────────────────────────────────────────────────────────────────────
[`fetch_boston`][7]       │Load the boston housing dataset (regression).                            
──────────────────────────┼─────────────────────────────────────────────────────────────────────────
[`fetch_credit_card`][8]  │Load the 'Default of Credit Card clients' dataset (binary                
                          │classification).                                                         
──────────────────────────┼─────────────────────────────────────────────────────────────────────────
[`fetch_diabetes_hospital`│Load the preprocessed Diabetes 130-Hospitals dataset (binary             
][9]                      │classification).                                                         
──────────────────────────┴─────────────────────────────────────────────────────────────────────────

# [`fairlearn.metrics`][10][#][11]

Functionality for computing metrics, with a particular focus on disaggregated metrics.

For our purpose, a metric is a function with signature `f(y_true, y_pred, ....)` where `y_true` are
the set of true values and `y_pred` are values predicted by a machine learning algorithm. Other
arguments may be present (most often sample weights), which will affect how the metric is
calculated.

This module provides the concept of a *disaggregated metric*. This is a metric where in addition to
`y_true` and `y_pred` values, the user provides information about group membership for each sample.
For example, a user could provide a ‘Gender’ column, and the disaggregated metric would contain
separate results for the subgroups ‘male’, ‘female’ and ‘nonbinary’ indicated by that column. The
underlying metric function is evaluated for each of these three subgroups. This extends to multiple
grouping columns, calculating the metric for each combination of subgroups.

───────────────────────────┬────────────────────────────────────────────────────────────────────────
[`count`][12]              │Calculate the number of data points in each group when working with     
                           │MetricFrame.                                                            
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`demographic_parity_differ│Calculate the demographic parity difference.                            
ence`][13]                 │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`demographic_parity_ratio`│Calculate the demographic parity ratio.                                 
][14]                      │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`equalized_odds_difference│Calculate the equalized odds difference.                                
`][15]                     │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`equalized_odds_ratio`][16│Calculate the equalized odds ratio.                                     
]                          │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`equal_opportunity_differe│Calculate the equal opportunity difference.                             
nce`][17]                  │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`equal_opportunity_ratio`]│Calculate the equal opportunity ratio.                                  
[18]                       │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`false_negative_rate`][19]│Calculate the false negative rate (also called miss rate).              
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`false_positive_rate`][20]│Calculate the false positive rate (also called fall-out).               
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`make_derived_metric`][21]│Create a scalar returning metric function based on aggregation of a     
                           │disaggregated metric.                                                   
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`mean_prediction`][22]    │Calculate the (weighted) mean prediction.                               
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`plot_model_comparison`][2│Create a scatter plot comparing multiple models along two metrics.      
3]                         │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`selection_rate`][24]     │Calculate the fraction of predicted labels matching the 'good' outcome. 
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`true_negative_rate`][25] │Calculate the true negative rate (also called specificity or            
                           │selectivity).                                                           
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`true_positive_rate`][26] │Calculate the true positive rate (also called sensitivity, recall, or   
                           │hit rate).                                                              
───────────────────────────┴────────────────────────────────────────────────────────────────────────

────────────────┬──────────────────────────────────────────
[`MetricFrame`][│Collection of disaggregated metric values.
27]             │                                          
────────────────┴──────────────────────────────────────────

# [`fairlearn.postprocessing`][28][#][29]

This module contains methods which operate on a predictor, rather than an estimator.

The predictor’s output is adjusted to fulfill specified parity constraints. The postprocessors learn
how to adjust the predictor’s output from the training data.

───────────────────────┬──────────────────────────────────────────────────────────
[`ThresholdOptimizer`][│A classifier based on the threshold optimization approach.
30]                    │                                                          
───────────────────────┴──────────────────────────────────────────────────────────

─────────────────────────────┬────────────────────────────────────────────────────
[`plot_threshold_optimizer`][│Plot the chosen solution of the threshold optimizer.
31]                          │                                                    
─────────────────────────────┴────────────────────────────────────────────────────

# [`fairlearn.preprocessing`][32][#][33]

Preprocessing tools to help deal with sensitive attributes.

───────────────┬────────────────────────────────────────────────────────────────────────────────────
[`CorrelationRe│A component that filters out sensitive correlations in a dataset.                   
mover`][34]    │                                                                                    
───────────────┼────────────────────────────────────────────────────────────────────────────────────
[`PrototypeRepr│A transformer and classifier that learns a latent representation of the input data  
esentationLearn│to obfuscate the sensitive features while preserving the classification and         
er`][35]       │reconstruction performance.                                                         
───────────────┴────────────────────────────────────────────────────────────────────────────────────

# [`fairlearn.reductions`][36][#][37]

This module contains algorithms implementing the reductions approach to disparity mitigation.

In this approach, disparity constraints are cast as Lagrange multipliers, which cause the
reweighting and relabelling of the input data. This *reduces* the problem back to standard machine
learning training.

───────────────────────────┬────────────────────────────────────────────────────────────────────────
[`AbsoluteLoss`][38]       │Class to evaluate absolute loss.                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`BoundedGroupLoss`][39]   │Moment for constraining the worst-case loss by a group.                 
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`ClassificationMoment`][40│Moment that can be expressed as weighted classification error.          
]                          │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`DemographicParity`][41]  │Implementation of demographic parity as a moment.                       
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`EqualizedOdds`][42]      │Implementation of equalized odds as a moment.                           
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`ErrorRate`][43]          │Misclassification error as a moment.                                    
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`ErrorRateParity`][44]    │Implementation of error rate parity as a moment.                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`ExponentiatedGradient`][4│An Estimator which implements the exponentiated gradient reduction.     
5]                         │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`TruePositiveRateParity`][│Implementation of true positive rate parity as a moment.                
46]                        │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`FalsePositiveRateParity`]│Implementation of false positive rate parity as a moment.               
[47]                       │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`UtilityParity`][48]      │A generic moment for parity in utilities (or costs) under               
                           │classification.                                                         
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`GridSearch`][49]         │Estimator to perform a grid search given a blackbox estimator algorithm.
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LossMoment`][50]         │Moment that can be expressed as weighted loss.                          
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`Moment`][51]             │Generic moment.                                                         
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`BoundedGroupLoss`][52]   │Moment for constraining the worst-case loss by a group.                 
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`SquareLoss`][53]         │Class to evaluate the square loss.                                      
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`ZeroOneLoss`][54]        │Class to evaluate a zero-one loss.                                      
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`MeanLoss`][55]           │Moment for evaluating the mean loss.                                    
───────────────────────────┴────────────────────────────────────────────────────────────────────────

# [`fairlearn.adversarial`][56][#][57]

Adversarial techniques to help mitigate unfairness.

─────────────────────────────────┬──────────────────────────────────────────────────────────────────
[`AdversarialFairnessClassifier`]│Train PyTorch or TensorFlow classifiers while mitigating          
[58]                             │unfairness.                                                       
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`AdversarialFairnessRegressor`][│Train PyTorch or TensorFlow regressors while mitigating           
59]                              │unfairness.                                                       
─────────────────────────────────┴──────────────────────────────────────────────────────────────────

# [`fairlearn.experimental`][60][#][61]

Enables experimental functionality that may be migrated to other modules at a later point.

Warning

Anything can break from version to version without further warning.

─────────────────────────────────────┬──────────────────────────────────────────────────────────────
[`metrics._plotter.plot_metric_frame`│Visualization for metrics with and without confidence         
][62]                                │intervals.                                                    
─────────────────────────────────────┴──────────────────────────────────────────────────────────────

[1]: #api-docs
[2]: #module-fairlearn.datasets
[3]: #module-fairlearn.datasets
[4]: generated/fairlearn.datasets.fetch_acs_income.html#fairlearn.datasets.fetch_acs_income
[5]: generated/fairlearn.datasets.fetch_adult.html#fairlearn.datasets.fetch_adult
[6]: generated/fairlearn.datasets.fetch_bank_marketing.html#fairlearn.datasets.fetch_bank_marketing
[7]: generated/fairlearn.datasets.fetch_boston.html#fairlearn.datasets.fetch_boston
[8]: generated/fairlearn.datasets.fetch_credit_card.html#fairlearn.datasets.fetch_credit_card
[9]: generated/fairlearn.datasets.fetch_diabetes_hospital.html#fairlearn.datasets.fetch_diabetes_hos
pital
[10]: #module-fairlearn.metrics
[11]: #module-fairlearn.metrics
[12]: generated/fairlearn.metrics.count.html#fairlearn.metrics.count
[13]: generated/fairlearn.metrics.demographic_parity_difference.html#fairlearn.metrics.demographic_p
arity_difference
[14]: generated/fairlearn.metrics.demographic_parity_ratio.html#fairlearn.metrics.demographic_parity
_ratio
[15]: generated/fairlearn.metrics.equalized_odds_difference.html#fairlearn.metrics.equalized_odds_di
fference
[16]: generated/fairlearn.metrics.equalized_odds_ratio.html#fairlearn.metrics.equalized_odds_ratio
[17]: generated/fairlearn.metrics.equal_opportunity_difference.html#fairlearn.metrics.equal_opportun
ity_difference
[18]: generated/fairlearn.metrics.equal_opportunity_ratio.html#fairlearn.metrics.equal_opportunity_r
atio
[19]: generated/fairlearn.metrics.false_negative_rate.html#fairlearn.metrics.false_negative_rate
[20]: generated/fairlearn.metrics.false_positive_rate.html#fairlearn.metrics.false_positive_rate
[21]: generated/fairlearn.metrics.make_derived_metric.html#fairlearn.metrics.make_derived_metric
[22]: generated/fairlearn.metrics.mean_prediction.html#fairlearn.metrics.mean_prediction
[23]: generated/fairlearn.metrics.plot_model_comparison.html#fairlearn.metrics.plot_model_comparison
[24]: generated/fairlearn.metrics.selection_rate.html#fairlearn.metrics.selection_rate
[25]: generated/fairlearn.metrics.true_negative_rate.html#fairlearn.metrics.true_negative_rate
[26]: generated/fairlearn.metrics.true_positive_rate.html#fairlearn.metrics.true_positive_rate
[27]: generated/fairlearn.metrics.MetricFrame.html#fairlearn.metrics.MetricFrame
[28]: #module-fairlearn.postprocessing
[29]: #module-fairlearn.postprocessing
[30]: generated/fairlearn.postprocessing.ThresholdOptimizer.html#fairlearn.postprocessing.ThresholdO
ptimizer
[31]: generated/fairlearn.postprocessing.plot_threshold_optimizer.html#fairlearn.postprocessing.plot
_threshold_optimizer
[32]: #module-fairlearn.preprocessing
[33]: #module-fairlearn.preprocessing
[34]: generated/fairlearn.preprocessing.CorrelationRemover.html#fairlearn.preprocessing.CorrelationR
emover
[35]: generated/fairlearn.preprocessing.PrototypeRepresentationLearner.html#fairlearn.preprocessing.
PrototypeRepresentationLearner
[36]: #module-fairlearn.reductions
[37]: #module-fairlearn.reductions
[38]: generated/fairlearn.reductions.AbsoluteLoss.html#fairlearn.reductions.AbsoluteLoss
[39]: generated/fairlearn.reductions.BoundedGroupLoss.html#fairlearn.reductions.BoundedGroupLoss
[40]: generated/fairlearn.reductions.ClassificationMoment.html#fairlearn.reductions.ClassificationMo
ment
[41]: generated/fairlearn.reductions.DemographicParity.html#fairlearn.reductions.DemographicParity
[42]: generated/fairlearn.reductions.EqualizedOdds.html#fairlearn.reductions.EqualizedOdds
[43]: generated/fairlearn.reductions.ErrorRate.html#fairlearn.reductions.ErrorRate
[44]: generated/fairlearn.reductions.ErrorRateParity.html#fairlearn.reductions.ErrorRateParity
[45]: generated/fairlearn.reductions.ExponentiatedGradient.html#fairlearn.reductions.ExponentiatedGr
adient
[46]: generated/fairlearn.reductions.TruePositiveRateParity.html#fairlearn.reductions.TruePositiveRa
teParity
[47]: generated/fairlearn.reductions.FalsePositiveRateParity.html#fairlearn.reductions.FalsePositive
RateParity
[48]: generated/fairlearn.reductions.UtilityParity.html#fairlearn.reductions.UtilityParity
[49]: generated/fairlearn.reductions.GridSearch.html#fairlearn.reductions.GridSearch
[50]: generated/fairlearn.reductions.LossMoment.html#fairlearn.reductions.LossMoment
[51]: generated/fairlearn.reductions.Moment.html#fairlearn.reductions.Moment
[52]: generated/fairlearn.reductions.BoundedGroupLoss.html#fairlearn.reductions.BoundedGroupLoss
[53]: generated/fairlearn.reductions.SquareLoss.html#fairlearn.reductions.SquareLoss
[54]: generated/fairlearn.reductions.ZeroOneLoss.html#fairlearn.reductions.ZeroOneLoss
[55]: generated/fairlearn.reductions.MeanLoss.html#fairlearn.reductions.MeanLoss
[56]: #module-fairlearn.adversarial
[57]: #module-fairlearn.adversarial
[58]: generated/fairlearn.adversarial.AdversarialFairnessClassifier.html#fairlearn.adversarial.Adver
sarialFairnessClassifier
[59]: generated/fairlearn.adversarial.AdversarialFairnessRegressor.html#fairlearn.adversarial.Advers
arialFairnessRegressor
[60]: #module-fairlearn.experimental
[61]: #module-fairlearn.experimental
[62]: generated/fairlearn.metrics._plotter.plot_metric_frame.html#fairlearn.metrics._plotter.plot_me
tric_frame
