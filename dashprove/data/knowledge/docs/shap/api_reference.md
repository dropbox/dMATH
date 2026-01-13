* API Reference
* [ View page source][1]

# API Reference[][2]

This page contains the API reference for public objects and functions in SHAP. There are also
[example notebooks][3] available that demonstrate how to use the API of each object/function.

## Explanation[][4]

─────────────────────────────────┬──────────────────────────────────────────────────────────────────
[`shap.Explanation`][5](values[, │A sliceable set of parallel arrays representing a SHAP            
base_values, ...])               │explanation.                                                      
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.Cohorts`][6](**kwargs)    │A collection of [`Explanation`][7] objects, typically each        
                                 │explaining a cluster of similar samples.                          
─────────────────────────────────┴──────────────────────────────────────────────────────────────────

## explainers[][8]

────────────────────────────────┬───────────────────────────────────────────────────────────────────
[`shap.Explainer`][9](model,    │Uses Shapley values to explain any machine learning model or python
masker, link, ...)              │function.                                                          
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.TreeExplainer`][10](model│Uses Tree SHAP algorithms to explain the output of ensemble tree   
, data, ...)                    │models.                                                            
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.GPUTreeExplainer`][11](mo│Experimental GPU accelerated version of TreeExplainer.             
del, data, ...)                 │                                                                   
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.LinearExplainer`][12](mod│Computes SHAP values for a linear model, optionally accounting for 
el, masker, link, ...)          │inter-feature correlations.                                        
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.PermutationExplainer`][13│This method approximates the Shapley values by iterating through   
](model, masker, ...)           │permutations of the inputs.                                        
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.PartitionExplainer`][14](│Uses the Partition SHAP method to explain the output of any        
model, masker, *, ...)          │function.                                                          
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.SamplingExplainer`][15](m│Computes SHAP values using an extension of the Shapley sampling    
odel, data, **kwargs)           │values explanation method (also known as IME).                     
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.AdditiveExplainer`][16](m│Computes SHAP values for generalized additive models.              
odel, masker[, ...])            │                                                                   
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.DeepExplainer`][17](model│Meant to approximate SHAP values for deep learning models.         
, data[, session, ...])         │                                                                   
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.KernelExplainer`][18](mod│Uses the Kernel SHAP method to explain the output of any function. 
el, data[, ...])                │                                                                   
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.GradientExplainer`][19](m│Explains a model using expected gradients (an extension of         
odel, data[, ...])              │integrated gradients).                                             
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.ExactExplainer`][20](mode│Computes SHAP values via an optimized exact enumeration.           
l, masker, link, ...)           │                                                                   
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.explainers.other.Coeffici│Simply returns the model coefficients as the feature attributions. 
ent`][21](model)                │                                                                   
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.explainers.other.Random`]│Simply returns random (normally distributed) feature attributions. 
[22](model, masker)             │                                                                   
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.explainers.other.LimeTabu│Simply wrap of lime.lime_tabular.LimeTabularExplainer into the     
lar`][23](model, data)          │common shap interface.                                             
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.explainers.other.Maple`][│Simply wraps MAPLE into the common SHAP interface.                 
24](model, data)                │                                                                   
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.explainers.other.TreeMapl│Simply tree MAPLE into the common SHAP interface.                  
e`][25](model, data)            │                                                                   
────────────────────────────────┼───────────────────────────────────────────────────────────────────
[`shap.explainers.other.TreeGain│Simply returns the global gain/gini feature importances for tree   
`][26](model)                   │models.                                                            
────────────────────────────────┴───────────────────────────────────────────────────────────────────

## plots[][27]

───────────────────────────────────────┬────────────────────────────────────────────────────────────
[`shap.plots.bar`][28](shap_values[,   │Create a bar plot of a set of SHAP values.                  
max_display, ...])                     │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.waterfall`][29](shap_value│Plots an explanation of a single prediction as a waterfall  
s[, ...])                              │plot.                                                       
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.scatter`][30](shap_values[│Create a SHAP dependence scatter plot, optionally colored by
, color, ...])                         │an interaction feature.                                     
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.heatmap`][31](shap_values[│Create a heatmap plot of a set of SHAP values.              
, ...])                                │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.force`][32](base_value[,  │Visualize the given SHAP values with an additive force      
shap_values, ...])                     │layout.                                                     
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.text`][33](shap_values[,  │Plots an explanation of a string of text using coloring and 
...])                                  │interactive labels.                                         
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.image`][34](shap_values[, │Plots SHAP values for image inputs.                         
...])                                  │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.partial_dependence`][35](i│A basic partial dependence plot function.                   
nd, model, data)                       │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.decision`][36](base_value,│Visualize model decisions using cumulative SHAP values.     
shap_values)                           │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.embedding`][37](ind,      │Use the SHAP values as an embedding which we project to 2D  
shap_values[, ...])                    │for visualization.                                          
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.initjs`][38]()            │Initialize the necessary javascript libraries for           
                                       │interactive force plots.                                    
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.group_difference`][39](sha│This plots the difference in mean SHAP values between two   
p_values, ...)                         │groups.                                                     
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.image_to_text`][40](shap_v│Plots SHAP values for image inputs with test outputs.       
alues)                                 │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.monitoring`][41](ind,     │Create a SHAP monitoring plot.                              
shap_values, features)                 │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.beeswarm`][42](shap_values│Create a SHAP beeswarm plot, colored by feature values when 
[, ...])                               │they are provided.                                          
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.plots.violin`][43](shap_values[,│Create a SHAP violin plot, colored by feature values when   
features, ...])                        │they are provided.                                          
───────────────────────────────────────┴────────────────────────────────────────────────────────────

## maskers[][44]

─────────────────────────────────┬──────────────────────────────────────────────────────────────────
[`shap.maskers.Masker`][45]()    │This is the superclass of all maskers.                            
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.maskers.Independent`][46](│This masks out tabular features by integrating over the given     
data[, max_samples])             │background dataset.                                               
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.maskers.Partition`][47](da│This masks out tabular features by integrating over the given     
ta[, max_samples, ...])          │background dataset.                                               
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.maskers.Impute`][48](data[│This imputes the values of missing features using the values of   
, method])                       │the observed features.                                            
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.maskers.Fixed`][49]()     │This leaves the input unchanged during masking, and is used for   
                                 │things like scoring labels.                                       
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.maskers.Composite`][50](*m│This merges several maskers for different inputs together into a  
askers)                          │single composite masker.                                          
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.maskers.FixedComposite`][5│A masker that outputs both the masked data and the original data  
1](masker)                       │as a pair.                                                        
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.maskers.OutputComposite`][│A masker that is a combination of a masker and a model and outputs
52](masker, model)               │both masked args and the model's output.                          
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.maskers.Text`][53]([tokeni│This masks out tokens according to the given tokenizer.           
zer, mask_token, ...])           │                                                                  
─────────────────────────────────┼──────────────────────────────────────────────────────────────────
[`shap.maskers.Image`][54](mask_v│Masks out image regions with blurring or inpainting.              
alue[, shape])                   │                                                                  
─────────────────────────────────┴──────────────────────────────────────────────────────────────────

## models[][55]

───────────────────────────────────┬────────────────────────────────────────────────────────────────
[`shap.models.Model`][56]([model]) │This is the superclass of all models.                           
───────────────────────────────────┼────────────────────────────────────────────────────────────────
[`shap.models.TeacherForcing`][57](│Generates scores (log odds) for output text explanation         
model[, ...])                      │algorithms using Teacher Forcing technique.                     
───────────────────────────────────┼────────────────────────────────────────────────────────────────
[`shap.models.TextGeneration`][58](│Generates target sentence/ids using a base model.               
[model, ...])                      │                                                                
───────────────────────────────────┼────────────────────────────────────────────────────────────────
[`shap.models.TopKLM`][59](model,  │Generates scores (log odds) for the top-k tokens for            
tokenizer[, k, ...])               │Causal/Masked LM.                                               
───────────────────────────────────┼────────────────────────────────────────────────────────────────
[`shap.models.TransformersPipeline`│This wraps a transformers pipeline object for easy explanations.
][60](pipeline[, ...])             │                                                                
───────────────────────────────────┴────────────────────────────────────────────────────────────────

## utils[][61]

──────────────────────────────────┬─────────────────────────────────────────────────────────────────
[`shap.utils.hclust`][62](X[, y,  │Fit a hierarchical clustering model for features X relative to   
linkage, metric, ...])            │target variable y.                                               
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.hclust_ordering`][63]│A leaf ordering is under-defined, this picks the ordering that   
(X[, metric, ...])                │keeps nearby samples similar.                                    
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.partition_tree`][64](│                                                                 
X[, metric])                      │                                                                 
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.partition_tree_shuffl│Randomly shuffle the indexes in a way that is consistent with the
e`][65](indexes, ...)             │given partition tree.                                            
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.delta_minimization_or│                                                                 
der`][66](all_masks)              │                                                                 
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.approximate_interacti│Order other features by how much interaction they seem to have   
ons`][67](index, ...)             │with the feature at the given index.                             
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.potential_interaction│Order other features by how much interaction they seem to have   
s`][68](...)                      │with the feature at the given index.                             
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.sample`][69](X[,     │Performs sampling without replacement of the input data `X`.     
nsamples, random_state])          │                                                                 
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.shapley_coefficients`│                                                                 
][70](n)                          │                                                                 
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.convert_name`][71](in│                                                                 
d, shap_values, ...)              │                                                                 
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.OpChain`][72]([root_n│A way to represent a set of dot chained operations on an object  
ame])                             │without actually running them.                                   
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.show_progress`][73](i│                                                                 
terable[, total, ...])            │                                                                 
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.MaskedModel`][74](mod│This is a utility class that combines a model, a masker object,  
el, masker, link, ...)            │and a current input.                                             
──────────────────────────────────┼─────────────────────────────────────────────────────────────────
[`shap.utils.make_masks`][75](clus│Builds a sparse CSR mask matrix from the given clustering.       
ter_matrix)                       │                                                                 
──────────────────────────────────┴─────────────────────────────────────────────────────────────────

## datasets[][76]

───────────────────────────────────────┬────────────────────────────────────────────────────────────
[`shap.datasets.a1a`][77]([n_points])  │Return a sparse dataset in scipy csr matrix format.         
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.adult`][78]([display,  │Return the Adult census data in a structured format.        
n_points])                             │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.california`][79]([n_poi│Return the California housing data in a tabular format.     
nts])                                  │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.communitiesandcrime`][8│Predict the total number of violent crimes per 100K         
0]([n_points])                         │population.                                                 
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.corrgroups60`][81]([n_p│Correlated Groups (60 features)                             
oints])                                │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.diabetes`][82]([n_point│Return the diabetes data in a nice package.                 
s])                                    │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.imagenet50`][83]([resol│Return a set of 50 images representative of ImageNet images.
ution, n_points])                      │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.imdb`][84]([n_points]) │Return the classic IMDB sentiment analysis training data in 
                                       │a nice package.                                             
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.independentlinear60`][8│Independent Linear (60 features)                            
5]([n_points])                         │                                                            
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.iris`][86]()           │Return the classic Iris dataset in a convenient package.    
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.linnerud`][87]([n_point│Return the Linnerud dataset in a convenient package for     
s])                                    │multi-target regression.                                    
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.nhanesi`][88]([display,│Return a nicely packaged version of NHANES I data with      
n_points])                             │survival times as labels.                                   
───────────────────────────────────────┼────────────────────────────────────────────────────────────
[`shap.datasets.rank`][89]()           │Return ranking datasets from the LightGBM repository.       
───────────────────────────────────────┴────────────────────────────────────────────────────────────
[ Previous][90] [Next ][91]

© Copyright 2018, Scott Lundberg.

Built with [Sphinx][92] using a [theme][93] provided by [Read the Docs][94].

[1]: _sources/api.rst.txt
[2]: #api-reference
[3]: api_examples.html#api-examples
[4]: #explanation
[5]: generated/shap.Explanation.html#shap.Explanation
[6]: generated/shap.Cohorts.html#shap.Cohorts
[7]: generated/shap.Explanation.html#shap.Explanation
[8]: #explainers
[9]: generated/shap.Explainer.html#shap.Explainer
[10]: generated/shap.TreeExplainer.html#shap.TreeExplainer
[11]: generated/shap.GPUTreeExplainer.html#shap.GPUTreeExplainer
[12]: generated/shap.LinearExplainer.html#shap.LinearExplainer
[13]: generated/shap.PermutationExplainer.html#shap.PermutationExplainer
[14]: generated/shap.PartitionExplainer.html#shap.PartitionExplainer
[15]: generated/shap.SamplingExplainer.html#shap.SamplingExplainer
[16]: generated/shap.AdditiveExplainer.html#shap.AdditiveExplainer
[17]: generated/shap.DeepExplainer.html#shap.DeepExplainer
[18]: generated/shap.KernelExplainer.html#shap.KernelExplainer
[19]: generated/shap.GradientExplainer.html#shap.GradientExplainer
[20]: generated/shap.ExactExplainer.html#shap.ExactExplainer
[21]: generated/shap.explainers.other.Coefficient.html#shap.explainers.other.Coefficient
[22]: generated/shap.explainers.other.Random.html#shap.explainers.other.Random
[23]: generated/shap.explainers.other.LimeTabular.html#shap.explainers.other.LimeTabular
[24]: generated/shap.explainers.other.Maple.html#shap.explainers.other.Maple
[25]: generated/shap.explainers.other.TreeMaple.html#shap.explainers.other.TreeMaple
[26]: generated/shap.explainers.other.TreeGain.html#shap.explainers.other.TreeGain
[27]: #plots
[28]: generated/shap.plots.bar.html#shap.plots.bar
[29]: generated/shap.plots.waterfall.html#shap.plots.waterfall
[30]: generated/shap.plots.scatter.html#shap.plots.scatter
[31]: generated/shap.plots.heatmap.html#shap.plots.heatmap
[32]: generated/shap.plots.force.html#shap.plots.force
[33]: generated/shap.plots.text.html#shap.plots.text
[34]: generated/shap.plots.image.html#shap.plots.image
[35]: generated/shap.plots.partial_dependence.html#shap.plots.partial_dependence
[36]: generated/shap.plots.decision.html#shap.plots.decision
[37]: generated/shap.plots.embedding.html#shap.plots.embedding
[38]: generated/shap.plots.initjs.html#shap.plots.initjs
[39]: generated/shap.plots.group_difference.html#shap.plots.group_difference
[40]: generated/shap.plots.image_to_text.html#shap.plots.image_to_text
[41]: generated/shap.plots.monitoring.html#shap.plots.monitoring
[42]: generated/shap.plots.beeswarm.html#shap.plots.beeswarm
[43]: generated/shap.plots.violin.html#shap.plots.violin
[44]: #maskers
[45]: generated/shap.maskers.Masker.html#shap.maskers.Masker
[46]: generated/shap.maskers.Independent.html#shap.maskers.Independent
[47]: generated/shap.maskers.Partition.html#shap.maskers.Partition
[48]: generated/shap.maskers.Impute.html#shap.maskers.Impute
[49]: generated/shap.maskers.Fixed.html#shap.maskers.Fixed
[50]: generated/shap.maskers.Composite.html#shap.maskers.Composite
[51]: generated/shap.maskers.FixedComposite.html#shap.maskers.FixedComposite
[52]: generated/shap.maskers.OutputComposite.html#shap.maskers.OutputComposite
[53]: generated/shap.maskers.Text.html#shap.maskers.Text
[54]: generated/shap.maskers.Image.html#shap.maskers.Image
[55]: #models
[56]: generated/shap.models.Model.html#shap.models.Model
[57]: generated/shap.models.TeacherForcing.html#shap.models.TeacherForcing
[58]: generated/shap.models.TextGeneration.html#shap.models.TextGeneration
[59]: generated/shap.models.TopKLM.html#shap.models.TopKLM
[60]: generated/shap.models.TransformersPipeline.html#shap.models.TransformersPipeline
[61]: #utils
[62]: generated/shap.utils.hclust.html#shap.utils.hclust
[63]: generated/shap.utils.hclust_ordering.html#shap.utils.hclust_ordering
[64]: generated/shap.utils.partition_tree.html#shap.utils.partition_tree
[65]: generated/shap.utils.partition_tree_shuffle.html#shap.utils.partition_tree_shuffle
[66]: generated/shap.utils.delta_minimization_order.html#shap.utils.delta_minimization_order
[67]: generated/shap.utils.approximate_interactions.html#shap.utils.approximate_interactions
[68]: generated/shap.utils.potential_interactions.html#shap.utils.potential_interactions
[69]: generated/shap.utils.sample.html#shap.utils.sample
[70]: generated/shap.utils.shapley_coefficients.html#shap.utils.shapley_coefficients
[71]: generated/shap.utils.convert_name.html#shap.utils.convert_name
[72]: generated/shap.utils.OpChain.html#shap.utils.OpChain
[73]: generated/shap.utils.show_progress.html#shap.utils.show_progress
[74]: generated/shap.utils.MaskedModel.html#shap.utils.MaskedModel
[75]: generated/shap.utils.make_masks.html#shap.utils.make_masks
[76]: #datasets
[77]: generated/shap.datasets.a1a.html#shap.datasets.a1a
[78]: generated/shap.datasets.adult.html#shap.datasets.adult
[79]: generated/shap.datasets.california.html#shap.datasets.california
[80]: generated/shap.datasets.communitiesandcrime.html#shap.datasets.communitiesandcrime
[81]: generated/shap.datasets.corrgroups60.html#shap.datasets.corrgroups60
[82]: generated/shap.datasets.diabetes.html#shap.datasets.diabetes
[83]: generated/shap.datasets.imagenet50.html#shap.datasets.imagenet50
[84]: generated/shap.datasets.imdb.html#shap.datasets.imdb
[85]: generated/shap.datasets.independentlinear60.html#shap.datasets.independentlinear60
[86]: generated/shap.datasets.iris.html#shap.datasets.iris
[87]: generated/shap.datasets.linnerud.html#shap.datasets.linnerud
[88]: generated/shap.datasets.nhanesi.html#shap.datasets.nhanesi
[89]: generated/shap.datasets.rank.html#shap.datasets.rank
[90]: example_notebooks/genomic_examples/DeepExplainer%20Genomics%20Example.html
[91]: generated/shap.Explanation.html
[92]: https://www.sphinx-doc.org/
[93]: https://github.com/readthedocs/sphinx_rtd_theme
[94]: https://readthedocs.org
