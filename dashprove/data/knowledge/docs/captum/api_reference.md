[[Captum]][1]

* [Docs][2]
* [Tutorials][3]
* [API Reference][4]
* [GitHub][5]

# Captum API Reference[¶][6]

API Reference

* [Attribution][7]
  
  * [Integrated Gradients][8]
  * [Saliency][9]
  * [DeepLift][10]
  * [DeepLiftShap][11]
  * [GradientShap][12]
  * [Input X Gradient][13]
  * [Guided Backprop][14]
  * [Guided GradCAM][15]
  * [Deconvolution][16]
  * [Feature Ablation][17]
  * [Occlusion][18]
  * [Feature Permutation][19]
  * [Shapley Value Sampling][20]
  * [Lime][21]
  * [KernelShap][22]
  * [LRP][23]
* [LLM Attribution Classes][24]
  
  * [LLMAttribution][25]
  * [LLMGradientAttribution][26]
  * [LLMAttributionResult][27]
* [NoiseTunnel][28]
  
  * [`NoiseTunnel`][29]
* [Layer Attribution][30]
  
  * [Layer Conductance][31]
  * [Layer Activation][32]
  * [Internal Influence][33]
  * [Layer Gradient X Activation][34]
  * [GradCAM][35]
  * [Layer DeepLift][36]
  * [Layer DeepLiftShap][37]
  * [Layer GradientShap][38]
  * [Layer Integrated Gradients][39]
  * [Layer Feature Ablation][40]
  * [Layer Feature Permutation][41]
  * [Layer LRP][42]
* [Neuron Attribution][43]
  
  * [Neuron Gradient][44]
  * [Neuron Integrated Gradients][45]
  * [Neuron Conductance][46]
  * [Neuron DeepLift][47]
  * [Neuron DeepLiftShap][48]
  * [Neuron GradientShap][49]
  * [Neuron Guided Backprop][50]
  * [Neuron Deconvolution][51]
  * [Neuron Feature Ablation][52]
* [Metrics][53]
  
  * [Infidelity][54]
  * [Sensitivity][55]
* [Robustness][56]
  
  * [FGSM][57]
  * [PGD][58]
  * [Attack Comparator][59]
  * [Min Param Perturbation][60]
* [Concept-based Interpretability][61]
  
  * [TCAV][62]
  * [ConceptInterpreter][63]
  * [Concept][64]
  * [Classifier][65]
* [Influential Examples][66]
  
  * [DataInfluence][67]
  * [SimilarityInfluence][68]
  * [TracInCPBase][69]
  * [TracInCP][70]
  * [TracInCPFast][71]
  * [TracInCPFastRandProj][72]
* [Module][73]
  
  * [BinaryConcreteStochasticGates][74]
  * [GaussianStochasticGates][75]
* [Utilities][76]
  
  * [Interpretable Input][77]
  * [Visualization][78]
  * [Interpretable Embeddings][79]
  * [Token Reference Base][80]
  * [Linear Models][81]
  * [Baselines][82]
* [Base Classes][83]
  
  * [Attribution][84]
  * [Layer Attribution][85]
  * [Neuron Attribution][86]
  * [Gradient Attribution][87]
  * [Perturbation Attribution][88]

Insights API Reference

* [Insights][89]
  
  * [Batch][90]
  * [AttributionVisualizer][91]
* [Features][92]
  
  * [BaseFeature][93]
  * [GeneralFeature][94]
  * [TextFeature][95]
  * [ImageFeature][96]

# Indices and Tables[¶][97]

* [Index][98]
* [Module Index][99]
* [Search Page][100]

# [Captum][101]

### Navigation

API Reference

* [Attribution][102]
* [LLM Attribution Classes][103]
* [NoiseTunnel][104]
* [Layer Attribution][105]
* [Neuron Attribution][106]
* [Metrics][107]
* [Robustness][108]
* [Concept-based Interpretability][109]
* [Influential Examples][110]
* [Module][111]
* [Utilities][112]
* [Base Classes][113]

Insights API Reference

* [Insights][114]
* [Features][115]

### Related Topics

* [Documentation overview][116]
  
  * Next: [Attribution][117]
Docs[Introduction][118][Getting Started][119][Tutorials][120][API Reference][121]
Legal[Privacy][122][Terms][123]
Social
[captum][124]
[[Facebook Open Source]][125] Copyright © 2025 Facebook Inc.

[1]: /
[2]: /docs/introduction
[3]: /tutorials/
[4]: /api/
[5]: https://github.com/pytorch/captum
[6]: #captum-api-reference
[7]: attribution.html
[8]: integrated_gradients.html
[9]: saliency.html
[10]: deep_lift.html
[11]: deep_lift_shap.html
[12]: gradient_shap.html
[13]: input_x_gradient.html
[14]: guided_backprop.html
[15]: guided_grad_cam.html
[16]: deconvolution.html
[17]: feature_ablation.html
[18]: occlusion.html
[19]: feature_permutation.html
[20]: shapley_value_sampling.html
[21]: lime.html
[22]: kernel_shap.html
[23]: lrp.html
[24]: llm_attr.html
[25]: llm_attr.html#llmattribution
[26]: llm_attr.html#llmgradientattribution
[27]: llm_attr.html#llmattributionresult
[28]: noise_tunnel.html
[29]: noise_tunnel.html#captum.attr.NoiseTunnel
[30]: layer.html
[31]: layer.html#layer-conductance
[32]: layer.html#layer-activation
[33]: layer.html#internal-influence
[34]: layer.html#layer-gradient-x-activation
[35]: layer.html#gradcam
[36]: layer.html#layer-deeplift
[37]: layer.html#layer-deepliftshap
[38]: layer.html#layer-gradientshap
[39]: layer.html#layer-integrated-gradients
[40]: layer.html#layer-feature-ablation
[41]: layer.html#layer-feature-permutation
[42]: layer.html#layer-lrp
[43]: neuron.html
[44]: neuron.html#neuron-gradient
[45]: neuron.html#neuron-integrated-gradients
[46]: neuron.html#neuron-conductance
[47]: neuron.html#neuron-deeplift
[48]: neuron.html#neuron-deepliftshap
[49]: neuron.html#neuron-gradientshap
[50]: neuron.html#neuron-guided-backprop
[51]: neuron.html#neuron-deconvolution
[52]: neuron.html#neuron-feature-ablation
[53]: metrics.html
[54]: metrics.html#infidelity
[55]: metrics.html#sensitivity
[56]: robust.html
[57]: robust.html#fgsm
[58]: robust.html#pgd
[59]: robust.html#attack-comparator
[60]: robust.html#min-param-perturbation
[61]: concept.html
[62]: concept.html#tcav
[63]: concept.html#conceptinterpreter
[64]: concept.html#concept
[65]: concept.html#classifier
[66]: influence.html
[67]: influence.html#datainfluence
[68]: influence.html#similarityinfluence
[69]: influence.html#tracincpbase
[70]: influence.html#tracincp
[71]: influence.html#tracincpfast
[72]: influence.html#tracincpfastrandproj
[73]: module.html
[74]: binary_concrete_stg.html
[75]: gaussian_stg.html
[76]: utilities.html
[77]: utilities.html#interpretable-input
[78]: utilities.html#visualization
[79]: utilities.html#interpretable-embeddings
[80]: utilities.html#token-reference-base
[81]: utilities.html#linear-models
[82]: utilities.html#baselines
[83]: base_classes.html
[84]: base_classes.html#attribution
[85]: base_classes.html#layer-attribution
[86]: base_classes.html#neuron-attribution
[87]: base_classes.html#gradient-attribution
[88]: base_classes.html#perturbation-attribution
[89]: insights.html
[90]: insights.html#batch
[91]: insights.html#attributionvisualizer
[92]: insights.html#features
[93]: insights.html#basefeature
[94]: insights.html#generalfeature
[95]: insights.html#textfeature
[96]: insights.html#imagefeature
[97]: #indices-and-tables
[98]: genindex.html
[99]: py-modindex.html
[100]: search.html
[101]: #
[102]: attribution.html
[103]: llm_attr.html
[104]: noise_tunnel.html
[105]: layer.html
[106]: neuron.html
[107]: metrics.html
[108]: robust.html
[109]: concept.html
[110]: influence.html
[111]: module.html
[112]: utilities.html
[113]: base_classes.html
[114]: insights.html
[115]: insights.html#features
[116]: #
[117]: attribution.html
[118]: /docs/introduction
[119]: /docs/getting_started
[120]: /tutorials/
[121]: /api/
[122]: https://opensource.facebook.com/legal/privacy/
[123]: https://opensource.facebook.com/legal/terms/
[124]: https://github.com/pytorch/captum
[125]: https://opensource.facebook.com/
