# Welcome to the Adversarial Robustness Toolbox[¶][1]

[[ART Logo]][2]

Adversarial Robustness Toolbox (ART) is a Python library for Machine Learning Security. ART provides
tools that enable developers and researchers to evaluate, defend, certify and verify Machine
Learning models and applications against the adversarial threats of Evasion, Poisoning, Extraction,
and Inference. ART supports all popular machine learning frameworks (TensorFlow, Keras, PyTorch,
MXNet, scikit-learn, XGBoost, LightGBM, CatBoost, GPy, etc.), all data types (images, tables, audio,
video, etc.) and machine learning tasks (classification, object detection, generation,
certification, etc.).

[[ART Logo]][3] [[ART Logo]][4]

The code of ART is on [GitHub][5] and the Wiki contains overviews of implemented [attacks][6],
[defences][7] and [metrics][8].

The library is under continuous development. Feedback, bug reports and contributions are very
welcome!

## Supported Machine Learning Libraries[¶][9]

* TensorFlow (v1 and v2) ([https://www.tensorflow.org][10])
* Keras ([https://www.keras.io][11])
* PyTorch ([https://www.pytorch.org][12])
* MXNet ([https://mxnet.apache.org][13])
* Scikit-learn ([https://www.scikit-learn.org][14])
* XGBoost ([https://www.xgboost.ai][15])
* LightGBM ([https://lightgbm.readthedocs.io][16])
* CatBoost ([https://www.catboost.ai][17])
* GPy ([https://sheffieldml.github.io/GPy/][18])

User guide

* [Setup][19]
* [Examples][20]
* [Notebooks][21]

Modules

* [`art.attacks`][22]
  
  * [Base Class Attacks][23]
  * [Base Class Evasion Attacks][24]
  * [Base Class Poisoning Attacks][25]
  * [Base Class Extraction Attacks][26]
  * [Base Class Inference Attacks][27]
  * [Base Class Reconstruction Attacks][28]
* [`art.attacks.evasion`][29]
  
  * [Adversarial Patch][30]
  * [Adversarial Patch - Numpy][31]
  * [Adversarial Patch - PyTorch][32]
  * [Adversarial Patch - TensorFlowV2][33]
  * [Adversarial Texture - PyTorch][34]
  * [Auto Attack][35]
  * [Auto Projected Gradient Descent (Auto-PGD)][36]
  * [Auto Conjugate Gradient (Auto-CG)][37]
  * [Boundary Attack / Decision-Based Attack][38]
  * [Brendel and Bethge Attack][39]
  * [Carlini and Wagner L_0 Attack][40]
  * [Carlini and Wagner L_2 Attack][41]
  * [Carlini and Wagner L_inf Attack][42]
  * [Carlini and Wagner ASR Attack][43]
  * [Composite Adversarial Attack - PyTorch][44]
  * [Decision Tree Attack][45]
  * [DeepFool][46]
  * [DPatch][47]
  * [RobustDPatch][48]
  * [Elastic Net Attack][49]
  * [Fast Gradient Method (FGM)][50]
  * [Feature Adversaries - Numpy][51]
  * [Feature Adversaries - PyTorch][52]
  * [Feature Adversaries - TensorFlow][53]
  * [Frame Saliency Attack][54]
  * [Geometric Decision Based Attack][55]
  * [GRAPHITE - Blackbox][56]
  * [GRAPHITE - Whitebox - PyTorch][57]
  * [High Confidence Low Uncertainty Attack][58]
  * [HopSkipJump Attack][59]
  * [Imperceptible ASR Attack][60]
  * [Imperceptible ASR Attack - PyTorch][61]
  * [Basic Iterative Method (BIM)][62]
  * [Projected Gradient Descent (PGD)][63]
  * [Projected Gradient Descent (PGD) - Numpy][64]
  * [Projected Gradient Descent (PGD) - PyTorch][65]
  * [Projected Gradient Descent (PGD) - TensorFlowV2][66]
  * [LaserAttack][67]
  * [LowProFool][68]
  * [NewtonFool][69]
  * [Malware Gradient Descent - TensorFlow][70]
  * [Over The Air Flickering Attack - PyTorch][71]
  * [PixelAttack][72]
  * [ThresholdAttack][73]
  * [Jacobian Saliency Map Attack (JSMA)][74]
  * [Shadow Attack][75]
  * [ShapeShifter Attack][76]
  * [Sign-OPT Attack][77]
  * [Simple Black-box Adversarial Attack][78]
  * [Spatial Transformations Attack][79]
  * [Square Attack][80]
  * [Targeted Universal Perturbation Attack][81]
  * [Universal Perturbation Attack][82]
  * [Virtual Adversarial Method][83]
  * [Wasserstein Attack][84]
  * [Zeroth-Order Optimization (ZOO) Attack][85]
* [`art.attacks.extraction`][86]
  
  * [Copycat CNN][87]
  * [Functionally Equivalent Extraction][88]
  * [Knockoff Nets][89]
* [`art.attacks.inference.attribute_inference`][90]
  
  * [Attribute Inference Baseline][91]
  * [Attribute Inference Black-Box][92]
  * [Attribute Inference Membership][93]
  * [Attribute Inference Base Line True Label][94]
  * [Attribute Inference White-Box Lifestyle Decision-Tree][95]
  * [Attribute Inference White-Box Decision-Tree][96]
* [`art.attacks.inference.membership_inference`][97]
  
  * [Membership Inference Black-Box][98]
  * [Membership Inference Black-Box Rule-Based][99]
  * [Membership Inference Label-Only - Decision Boundary][100]
  * [Membership Inference Label-Only - Gap Attack][101]
  * [Shadow Models][102]
* [`art.attacks.inference.model_inversion`][103]
  
  * [Model Inversion MIFace][104]
* [`art.attacks.inference.reconstruction`][105]
  
  * [Database Reconstruction][106]
* [`art.attacks.poisoning`][107]
  
  * [Backdoor Attack DGM ReD][108]
  * [Backdoor Attack DGM Trail][109]
  * [Adversarial Embedding Attack][110]
  * [Backdoor Poisoning Attack][111]
  * [Hidden Trigger Backdoor Attack][112]
  * [Bullseye Polytope Attack][113]
  * [Clean Label Backdoor Attack][114]
  * [Feature Collision Attack][115]
  * [Gradient Matching Attack][116]
  * [Poisoning SVM Attack][117]
  * [Sleeper Agent Attack][118]
* [`art.defences`][119]
* [`art.defences.detector.evasion`][120]
  
  * [Base Class][121]
  * [Binary Input Detector][122]
  * [Binary Activation Detector][123]
  * [Subset Scanning Detector][124]
* [`art.defences.detector.poison`][125]
  
  * [Base Class][126]
  * [Activation Defence][127]
  * [Data Provenance Defense][128]
  * [Reject on Negative Impact (RONI) Defense][129]
  * [Spectral Signature Defense][130]
* [`art.defences.postprocessor`][131]
  
  * [Base Class Postprocessor][132]
  * [Class Labels][133]
  * [Gaussian Noise][134]
  * [High Confidence][135]
  * [Reverse Sigmoid][136]
  * [Rounded][137]
* [`art.defences.preprocessor`][138]
  
  * [Base Class Preprocessor][139]
  * [CutMix][140]
  * [CutMix - PyTorch][141]
  * [CutMix - TensorFlowV2][142]
  * [Cutout][143]
  * [Cutout - PyTorch][144]
  * [Cutout - TensorFlowV2][145]
  * [Feature Squeezing][146]
  * [Gaussian Data Augmentation][147]
  * [InverseGAN][148]
  * [DefenseGAN][149]
  * [JPEG Compression][150]
  * [Label Smoothing][151]
  * [Mixup][152]
  * [Mixup - PyTorch][153]
  * [Mixup - TensorFlowV2][154]
  * [Mp3 Compression][155]
  * [PixelDefend][156]
  * [Resample][157]
  * [Spatial Smoothing][158]
  * [Spatial Smoothing - PyTorch][159]
  * [Spatial Smoothing - TensorFlow v2][160]
  * [Thermometer Encoding][161]
  * [Total Variance Minimization][162]
  * [Video Compression][163]
* [`art.defences.trainer`][164]
  
  * [Base Class Trainer][165]
  * [Adversarial Training][166]
  * [Adversarial Training Madry PGD][167]
  * [Adversarial Training Adversarial Weight Perturbation (AWP) - PyTorch][168]
  * [Adversarial Training Oracle Aligned Adversarial Training (OAAT) - PyTorch][169]
  * [Adversarial Training TRADES - PyTorch][170]
  * [Base Class Adversarial Training Fast is Better than Free][171]
  * [Adversarial Training Fast is Better than Free - PyTorch][172]
  * [Adversarial Training Certified - PyTorch][173]
  * [Adversarial Training Certified Interval Bound Propagation - PyTorch][174]
  * [DP - InstaHide Training][175]
* [`art.defences.transformer.evasion`][176]
  
  * [Defensive Distillation][177]
* [`art.defences.transformer.poisoning`][178]
  
  * [Neural Cleanse][179]
  * [STRIP][180]
* [`art.estimators`][181]
  
  * [Base Class Estimator][182]
  * [Mixin Base Class Loss Gradients][183]
  * [Mixin Base Class Neural Networks][184]
  * [Mixin Base Class Decision Trees][185]
  * [Base Class KerasEstimator][186]
  * [Base Class MXEstimator][187]
  * [Base Class PyTorchEstimator][188]
  * [Base Class ScikitlearnEstimator][189]
  * [Base Class TensorFlowEstimator][190]
  * [Base Class TensorFlowV2Estimator][191]
* [`art.estimators.certification`][192]
* [`art.estimators.certification.deep_z`][193]
  
  * [PytorchDeepZ][194]
  * [Zono Dense Layer][195]
  * [Zono Bounds][196]
  * [Zono Convolution Layer][197]
  * [Zono ReLU][198]
* [`art.estimators.certification.interval`][199]
  
  * [PyTorchIBPClassifier][200]
  * [Interval Dense Layer][201]
  * [Interval Convolution Layer][202]
  * [Interval Flatten Layer][203]
  * [Interval ReLU][204]
  * [Interval Bounds][205]
* [`art.estimators.certification.randomized_smoothing`][206]
  
  * [Mixin Base Class Randomized Smoothing][207]
  * [PyTorch Randomized Smoothing Classifier][208]
  * [TensorFlow V2 Randomized Smoothing Classifier][209]
* [`art.estimators.classification`][210]
  
  * [Mixin Base Class Classifier][211]
  * [Mixin Base Class Class Gradients][212]
  * [BlackBox Classifier][213]
  * [BlackBox Classifier NeuralNetwork][214]
  * [Deep Partition Aggregation Classifier][215]
  * [Keras Classifier][216]
  * [MXNet Classifier][217]
  * [PyTorch Classifier][218]
  * [Query-Efficient Black-box Gradient Estimation Classifier][219]
  * [TensorFlow Classifier][220]
  * [TensorFlow v2 Classifier][221]
  * [Ensemble Classifier][222]
  * [Scikit-learn Classifier Classifier][223]
  * [GPy Gaussian Process Classifier][224]
* [`art.estimators.classification.scikitlearn`][225]
  
  * [Base Class Scikit-learn][226]
  * [Scikit-learn DecisionTreeClassifier Classifier][227]
  * [Scikit-learn ExtraTreeClassifier Classifier][228]
  * [Scikit-learn AdaBoostClassifier Classifier][229]
  * [Scikit-learn BaggingClassifier Classifier][230]
  * [Scikit-learn ExtraTreesClassifier Classifier][231]
  * [Scikit-learn GradientBoostingClassifier Classifier][232]
  * [Scikit-learn RandomForestClassifier Classifier][233]
  * [Scikit-learn LogisticRegression Classifier][234]
  * [Scikit-learn SVC Classifier][235]
* [`art.estimators.encoding`][236]
  
  * [Mixin Base Class Encoder][237]
  * [TensorFlow Encoder][238]
* [`art.estimators.gan`][239]
  
  * [TensorFlowV2 GAN][240]
* [`art.estimators.generation`][241]
  
  * [Mixin Base Class Generator][242]
  * [TensorFlow Generator][243]
  * [TensorFlow 2 Generator][244]
* [`art.estimators.object_detection`][245]
  
  * [Mixin Base Class Object Detector][246]
  * [Object Detector PyTorch][247]
  * [Object Detector PyTorch Faster-RCNN][248]
  * [Object Detector PyTorch YOLO][249]
  * [Object Detector TensorFlow Faster-RCNN][250]
* [`art.estimators.object_tracking`][251]
  
  * [Mixin Base Class Object Tracker][252]
  * [Object Tracker PyTorch GOTURN][253]
* [`art.estimators.poison_mitigation`][254]
  
  * [Keras Neural Cleanse Classifier][255]
  * [Mixin Base Class STRIP][256]
* [`art.estimators.regression`][257]
  
  * [Mixin Base Class Regressor][258]
* [`art.estimators.regression.scikitlearn`][259]
  
  * [Base Class Scikit-learn][260]
  * [Scikit-learn Decision Tree Regressor][261]
* [`art.estimators.speech_recognition`][262]
  
  * [Mixin Base Class Speech Recognizer][263]
  * [Speech Recognizer Deep Speech - PyTorch][264]
  * [Speech Recognizer Espresso - PyTorch][265]
  * [Speech Recognizer Lingvo ASR - TensorFlow][266]
* [`art.experimental.estimators`][267]
  
  * [Base Class JaxEstimator][268]
* [`art.experimental.estimators.classification`][269]
  
  * [JAX Classifier][270]
* [`art.evaluations`][271]
* [`art.metrics`][272]
  
  * [Clique Method Robustness Verification][273]
  * [Loss Sensitivity][274]
  * [Empirical Robustness][275]
  * [CLEVER][276]
  * [Wasserstein Distance][277]
  * [Pointwise Differential Training Privacy][278]
  * [SHAPr Membership Privacy Risk][279]
* [`art.preprocessing`][280]
* [`art.preprocessing.audio`][281]
  
  * [L-Filter][282]
  * [LFilter - PyTorch][283]
* [`art.preprocessing.expectation_over_transformation`][284]
  
  * [EOT Image Center Crop - PyTorch][285]
  * [EOT Image Rotation - TensorFlow V2][286]
  * [EOT Image Rotation - PyTorch][287]
  * [EOT Brightness - PyTorch][288]
  * [EOT Brightness - TensorFlow V2][289]
  * [EOT Contrast - PyTorch][290]
  * [EOT Contrast - TensorFlow V2][291]
  * [EOT Gaussian Noise - PyTorch][292]
  * [EOT Gaussian Noise - TensorFlow V2][293]
  * [EOT Shot Noise - PyTorch][294]
  * [EOT Shot Noise - TensorFlow V2][295]
  * [EOT Zoom Blur - PyTorch][296]
  * [EOT Zoom Blur - TensorFlow V2][297]
* [`art.preprocessing.standardisation_mean_std`][298]
  
  * [Standardisation Mean and Std][299]
  * [Standardisation Mean and Std - PyTorch][300]
  * [Standardisation Mean and Std - TensorFlow V2][301]
* [`art.data_generators`][302]
  
  * [Base Class][303]
  * [Framework-Specific Data Generators][304]
* [`art.exceptions`][305]
  
  * [EstimatorError][306]
* [`art.summary_writer`][307]
  
  * [Base Class SummaryWriter][308]
  * [Summary Writer Default][309]
* [`art.utils`][310]
  
  * [Deprecation Operations][311]
  * [Math Operations][312]
  * [Label Operations][313]
  * [Dataset Operations][314]
* [`tests.utils`][315]
  
  * [Test Base Classes][316]
  * [Trained Models for Unittests, MNIST][317]
  * [Trained Models for Unittests, Iris][318]
  * [Random Number Generators][319]

# Indices and Tables[¶][320]

* [Index][321]
* [Module Index][322]
* [Search Page][323]

# [Adversarial Robustness Toolbox][324]

### Navigation

User guide

* [Setup][325]
* [Examples][326]
* [Notebooks][327]

Modules

* [`art.attacks`][328]
* [`art.attacks.evasion`][329]
* [`art.attacks.extraction`][330]
* [`art.attacks.inference.attribute_inference`][331]
* [`art.attacks.inference.membership_inference`][332]
* [`art.attacks.inference.model_inversion`][333]
* [`art.attacks.inference.reconstruction`][334]
* [`art.attacks.poisoning`][335]
* [`art.defences`][336]
* [`art.defences.detector.evasion`][337]
* [`art.defences.detector.poison`][338]
* [`art.defences.postprocessor`][339]
* [`art.defences.preprocessor`][340]
* [`art.defences.trainer`][341]
* [`art.defences.transformer.evasion`][342]
* [`art.defences.transformer.poisoning`][343]
* [`art.estimators`][344]
* [`art.estimators.certification`][345]
* [`art.estimators.certification.deep_z`][346]
* [`art.estimators.certification.interval`][347]
* [`art.estimators.certification.randomized_smoothing`][348]
* [`art.estimators.classification`][349]
* [`art.estimators.classification.scikitlearn`][350]
* [`art.estimators.encoding`][351]
* [`art.estimators.gan`][352]
* [`art.estimators.generation`][353]
* [`art.estimators.object_detection`][354]
* [`art.estimators.object_tracking`][355]
* [`art.estimators.poison_mitigation`][356]
* [`art.estimators.regression`][357]
* [`art.estimators.regression.scikitlearn`][358]
* [`art.estimators.speech_recognition`][359]
* [`art.experimental.estimators`][360]
* [`art.experimental.estimators.classification`][361]
* [`art.evaluations`][362]
* [`art.metrics`][363]
* [`art.preprocessing`][364]
* [`art.preprocessing.audio`][365]
* [`art.preprocessing.expectation_over_transformation`][366]
* [`art.preprocessing.standardisation_mean_std`][367]
* [`art.data_generators`][368]
* [`art.exceptions`][369]
* [`art.summary_writer`][370]
* [`art.utils`][371]
* [`tests.utils`][372]

### Related Topics

* [Documentation overview][373]
  
  * Next: [Setup][374]

### Quick search

©2018, The Adversarial Robustness Toolbox (ART) Authors. | Powered by [Sphinx 7.2.6][375] &
[Alabaster 0.7.16][376] | [Page source][377]

[1]: #welcome-to-the-adversarial-robustness-toolbox
[2]: _images/art_lfai.png
[3]: _images/adversarial_threats_attacker.png
[4]: _images/adversarial_threats_art.png
[5]: https://github.com/Trusted-AI/adversarial-robustness-toolbox
[6]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Attacks
[7]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Defences
[8]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Metrics
[9]: #supported-machine-learning-libraries
[10]: https://www.tensorflow.org
[11]: https://www.keras.io
[12]: https://www.pytorch.org
[13]: https://mxnet.apache.org
[14]: https://www.scikit-learn.org
[15]: https://www.xgboost.ai
[16]: https://lightgbm.readthedocs.io
[17]: https://www.catboost.ai
[18]: https://sheffieldml.github.io/GPy/
[19]: guide/setup.html
[20]: guide/examples.html
[21]: guide/notebooks.html
[22]: modules/attacks.html
[23]: modules/attacks.html#base-class-attacks
[24]: modules/attacks.html#base-class-evasion-attacks
[25]: modules/attacks.html#base-class-poisoning-attacks
[26]: modules/attacks.html#base-class-extraction-attacks
[27]: modules/attacks.html#base-class-inference-attacks
[28]: modules/attacks.html#base-class-reconstruction-attacks
[29]: modules/attacks/evasion.html
[30]: modules/attacks/evasion.html#adversarial-patch
[31]: modules/attacks/evasion.html#adversarial-patch-numpy
[32]: modules/attacks/evasion.html#adversarial-patch-pytorch
[33]: modules/attacks/evasion.html#adversarial-patch-tensorflowv2
[34]: modules/attacks/evasion.html#adversarial-texture-pytorch
[35]: modules/attacks/evasion.html#auto-attack
[36]: modules/attacks/evasion.html#auto-projected-gradient-descent-auto-pgd
[37]: modules/attacks/evasion.html#auto-conjugate-gradient-auto-cg
[38]: modules/attacks/evasion.html#boundary-attack-decision-based-attack
[39]: modules/attacks/evasion.html#brendel-and-bethge-attack
[40]: modules/attacks/evasion.html#carlini-and-wagner-l-0-attack
[41]: modules/attacks/evasion.html#carlini-and-wagner-l-2-attack
[42]: modules/attacks/evasion.html#carlini-and-wagner-l-inf-attack
[43]: modules/attacks/evasion.html#carlini-and-wagner-asr-attack
[44]: modules/attacks/evasion.html#composite-adversarial-attack-pytorch
[45]: modules/attacks/evasion.html#decision-tree-attack
[46]: modules/attacks/evasion.html#deepfool
[47]: modules/attacks/evasion.html#dpatch
[48]: modules/attacks/evasion.html#robustdpatch
[49]: modules/attacks/evasion.html#elastic-net-attack
[50]: modules/attacks/evasion.html#fast-gradient-method-fgm
[51]: modules/attacks/evasion.html#feature-adversaries-numpy
[52]: modules/attacks/evasion.html#feature-adversaries-pytorch
[53]: modules/attacks/evasion.html#feature-adversaries-tensorflow
[54]: modules/attacks/evasion.html#frame-saliency-attack
[55]: modules/attacks/evasion.html#geometric-decision-based-attack
[56]: modules/attacks/evasion.html#graphite-blackbox
[57]: modules/attacks/evasion.html#graphite-whitebox-pytorch
[58]: modules/attacks/evasion.html#high-confidence-low-uncertainty-attack
[59]: modules/attacks/evasion.html#hopskipjump-attack
[60]: modules/attacks/evasion.html#imperceptible-asr-attack
[61]: modules/attacks/evasion.html#imperceptible-asr-attack-pytorch
[62]: modules/attacks/evasion.html#basic-iterative-method-bim
[63]: modules/attacks/evasion.html#projected-gradient-descent-pgd
[64]: modules/attacks/evasion.html#projected-gradient-descent-pgd-numpy
[65]: modules/attacks/evasion.html#projected-gradient-descent-pgd-pytorch
[66]: modules/attacks/evasion.html#projected-gradient-descent-pgd-tensorflowv2
[67]: modules/attacks/evasion.html#laserattack
[68]: modules/attacks/evasion.html#lowprofool
[69]: modules/attacks/evasion.html#newtonfool
[70]: modules/attacks/evasion.html#malware-gradient-descent-tensorflow
[71]: modules/attacks/evasion.html#over-the-air-flickering-attack-pytorch
[72]: modules/attacks/evasion.html#pixelattack
[73]: modules/attacks/evasion.html#thresholdattack
[74]: modules/attacks/evasion.html#jacobian-saliency-map-attack-jsma
[75]: modules/attacks/evasion.html#shadow-attack
[76]: modules/attacks/evasion.html#shapeshifter-attack
[77]: modules/attacks/evasion.html#sign-opt-attack
[78]: modules/attacks/evasion.html#simple-black-box-adversarial-attack
[79]: modules/attacks/evasion.html#spatial-transformations-attack
[80]: modules/attacks/evasion.html#square-attack
[81]: modules/attacks/evasion.html#targeted-universal-perturbation-attack
[82]: modules/attacks/evasion.html#universal-perturbation-attack
[83]: modules/attacks/evasion.html#virtual-adversarial-method
[84]: modules/attacks/evasion.html#wasserstein-attack
[85]: modules/attacks/evasion.html#zeroth-order-optimization-zoo-attack
[86]: modules/attacks/extraction.html
[87]: modules/attacks/extraction.html#copycat-cnn
[88]: modules/attacks/extraction.html#functionally-equivalent-extraction
[89]: modules/attacks/extraction.html#knockoff-nets
[90]: modules/attacks/inference/attribute_inference.html
[91]: modules/attacks/inference/attribute_inference.html#attribute-inference-baseline
[92]: modules/attacks/inference/attribute_inference.html#attribute-inference-black-box
[93]: modules/attacks/inference/attribute_inference.html#attribute-inference-membership
[94]: modules/attacks/inference/attribute_inference.html#attribute-inference-base-line-true-label
[95]: modules/attacks/inference/attribute_inference.html#attribute-inference-white-box-lifestyle-dec
ision-tree
[96]: modules/attacks/inference/attribute_inference.html#attribute-inference-white-box-decision-tree
[97]: modules/attacks/inference/membership_inference.html
[98]: modules/attacks/inference/membership_inference.html#membership-inference-black-box
[99]: modules/attacks/inference/membership_inference.html#membership-inference-black-box-rule-based
[100]: modules/attacks/inference/membership_inference.html#membership-inference-label-only-decision-
boundary
[101]: modules/attacks/inference/membership_inference.html#membership-inference-label-only-gap-attac
k
[102]: modules/attacks/inference/membership_inference.html#shadow-models
[103]: modules/attacks/inference/model_inversion.html
[104]: modules/attacks/inference/model_inversion.html#model-inversion-miface
[105]: modules/attacks/inference/reconstruction.html
[106]: modules/attacks/inference/reconstruction.html#database-reconstruction
[107]: modules/attacks/poisoning.html
[108]: modules/attacks/poisoning.html#backdoor-attack-dgm-red
[109]: modules/attacks/poisoning.html#backdoor-attack-dgm-trail
[110]: modules/attacks/poisoning.html#adversarial-embedding-attack
[111]: modules/attacks/poisoning.html#backdoor-poisoning-attack
[112]: modules/attacks/poisoning.html#hidden-trigger-backdoor-attack
[113]: modules/attacks/poisoning.html#bullseye-polytope-attack
[114]: modules/attacks/poisoning.html#clean-label-backdoor-attack
[115]: modules/attacks/poisoning.html#feature-collision-attack
[116]: modules/attacks/poisoning.html#gradient-matching-attack
[117]: modules/attacks/poisoning.html#poisoning-svm-attack
[118]: modules/attacks/poisoning.html#sleeper-agent-attack
[119]: modules/defences.html
[120]: modules/defences/detector_evasion.html
[121]: modules/defences/detector_evasion.html#base-class
[122]: modules/defences/detector_evasion.html#binary-input-detector
[123]: modules/defences/detector_evasion.html#binary-activation-detector
[124]: modules/defences/detector_evasion.html#subset-scanning-detector
[125]: modules/defences/detector_poisoning.html
[126]: modules/defences/detector_poisoning.html#base-class
[127]: modules/defences/detector_poisoning.html#activation-defence
[128]: modules/defences/detector_poisoning.html#data-provenance-defense
[129]: modules/defences/detector_poisoning.html#reject-on-negative-impact-roni-defense
[130]: modules/defences/detector_poisoning.html#spectral-signature-defense
[131]: modules/defences/postprocessor.html
[132]: modules/defences/postprocessor.html#base-class-postprocessor
[133]: modules/defences/postprocessor.html#class-labels
[134]: modules/defences/postprocessor.html#gaussian-noise
[135]: modules/defences/postprocessor.html#high-confidence
[136]: modules/defences/postprocessor.html#reverse-sigmoid
[137]: modules/defences/postprocessor.html#rounded
[138]: modules/defences/preprocessor.html
[139]: modules/defences/preprocessor.html#base-class-preprocessor
[140]: modules/defences/preprocessor.html#cutmix
[141]: modules/defences/preprocessor.html#cutmix-pytorch
[142]: modules/defences/preprocessor.html#cutmix-tensorflowv2
[143]: modules/defences/preprocessor.html#cutout
[144]: modules/defences/preprocessor.html#cutout-pytorch
[145]: modules/defences/preprocessor.html#cutout-tensorflowv2
[146]: modules/defences/preprocessor.html#feature-squeezing
[147]: modules/defences/preprocessor.html#gaussian-data-augmentation
[148]: modules/defences/preprocessor.html#inversegan
[149]: modules/defences/preprocessor.html#defensegan
[150]: modules/defences/preprocessor.html#jpeg-compression
[151]: modules/defences/preprocessor.html#label-smoothing
[152]: modules/defences/preprocessor.html#mixup
[153]: modules/defences/preprocessor.html#mixup-pytorch
[154]: modules/defences/preprocessor.html#mixup-tensorflowv2
[155]: modules/defences/preprocessor.html#mp3-compression
[156]: modules/defences/preprocessor.html#pixeldefend
[157]: modules/defences/preprocessor.html#resample
[158]: modules/defences/preprocessor.html#spatial-smoothing
[159]: modules/defences/preprocessor.html#spatial-smoothing-pytorch
[160]: modules/defences/preprocessor.html#spatial-smoothing-tensorflow-v2
[161]: modules/defences/preprocessor.html#thermometer-encoding
[162]: modules/defences/preprocessor.html#total-variance-minimization
[163]: modules/defences/preprocessor.html#video-compression
[164]: modules/defences/trainer.html
[165]: modules/defences/trainer.html#base-class-trainer
[166]: modules/defences/trainer.html#adversarial-training
[167]: modules/defences/trainer.html#adversarial-training-madry-pgd
[168]: modules/defences/trainer.html#adversarial-training-adversarial-weight-perturbation-awp-pytorc
h
[169]: modules/defences/trainer.html#adversarial-training-oracle-aligned-adversarial-training-oaat-p
ytorch
[170]: modules/defences/trainer.html#adversarial-training-trades-pytorch
[171]: modules/defences/trainer.html#base-class-adversarial-training-fast-is-better-than-free
[172]: modules/defences/trainer.html#adversarial-training-fast-is-better-than-free-pytorch
[173]: modules/defences/trainer.html#adversarial-training-certified-pytorch
[174]: modules/defences/trainer.html#adversarial-training-certified-interval-bound-propagation-pytor
ch
[175]: modules/defences/trainer.html#dp-instahide-training
[176]: modules/defences/transformer_evasion.html
[177]: modules/defences/transformer_evasion.html#defensive-distillation
[178]: modules/defences/transformer_poisoning.html
[179]: modules/defences/transformer_poisoning.html#neural-cleanse
[180]: modules/defences/transformer_poisoning.html#strip
[181]: modules/estimators.html
[182]: modules/estimators.html#base-class-estimator
[183]: modules/estimators.html#mixin-base-class-loss-gradients
[184]: modules/estimators.html#mixin-base-class-neural-networks
[185]: modules/estimators.html#mixin-base-class-decision-trees
[186]: modules/estimators.html#base-class-kerasestimator
[187]: modules/estimators.html#base-class-mxestimator
[188]: modules/estimators.html#base-class-pytorchestimator
[189]: modules/estimators.html#base-class-scikitlearnestimator
[190]: modules/estimators.html#base-class-tensorflowestimator
[191]: modules/estimators.html#base-class-tensorflowv2estimator
[192]: modules/estimators/certification.html
[193]: modules/estimators/certification_deep_z.html
[194]: modules/estimators/certification_deep_z.html#pytorchdeepz
[195]: modules/estimators/certification_deep_z.html#zono-dense-layer
[196]: modules/estimators/certification_deep_z.html#zono-bounds
[197]: modules/estimators/certification_deep_z.html#zono-convolution-layer
[198]: modules/estimators/certification_deep_z.html#zono-relu
[199]: modules/estimators/certification_interval.html
[200]: modules/estimators/certification_interval.html#pytorchibpclassifier
[201]: modules/estimators/certification_interval.html#interval-dense-layer
[202]: modules/estimators/certification_interval.html#interval-convolution-layer
[203]: modules/estimators/certification_interval.html#interval-flatten-layer
[204]: modules/estimators/certification_interval.html#interval-relu
[205]: modules/estimators/certification_interval.html#interval-bounds
[206]: modules/estimators/certification_randomized_smoothing.html
[207]: modules/estimators/certification_randomized_smoothing.html#mixin-base-class-randomized-smooth
ing
[208]: modules/estimators/certification_randomized_smoothing.html#pytorch-randomized-smoothing-class
ifier
[209]: modules/estimators/certification_randomized_smoothing.html#tensorflow-v2-randomized-smoothing
-classifier
[210]: modules/estimators/classification.html
[211]: modules/estimators/classification.html#mixin-base-class-classifier
[212]: modules/estimators/classification.html#mixin-base-class-class-gradients
[213]: modules/estimators/classification.html#blackbox-classifier
[214]: modules/estimators/classification.html#blackbox-classifier-neuralnetwork
[215]: modules/estimators/classification.html#deep-partition-aggregation-classifier
[216]: modules/estimators/classification.html#keras-classifier
[217]: modules/estimators/classification.html#mxnet-classifier
[218]: modules/estimators/classification.html#pytorch-classifier
[219]: modules/estimators/classification.html#query-efficient-black-box-gradient-estimation-classifi
er
[220]: modules/estimators/classification.html#tensorflow-classifier
[221]: modules/estimators/classification.html#tensorflow-v2-classifier
[222]: modules/estimators/classification.html#ensemble-classifier
[223]: modules/estimators/classification.html#scikit-learn-classifier-classifier
[224]: modules/estimators/classification.html#gpy-gaussian-process-classifier
[225]: modules/estimators/classification_scikitlearn.html
[226]: modules/estimators/classification_scikitlearn.html#base-class-scikit-learn
[227]: modules/estimators/classification_scikitlearn.html#scikit-learn-decisiontreeclassifier-classi
fier
[228]: modules/estimators/classification_scikitlearn.html#scikit-learn-extratreeclassifier-classifie
r
[229]: modules/estimators/classification_scikitlearn.html#scikit-learn-adaboostclassifier-classifier
[230]: modules/estimators/classification_scikitlearn.html#scikit-learn-baggingclassifier-classifier
[231]: modules/estimators/classification_scikitlearn.html#scikit-learn-extratreesclassifier-classifi
er
[232]: modules/estimators/classification_scikitlearn.html#scikit-learn-gradientboostingclassifier-cl
assifier
[233]: modules/estimators/classification_scikitlearn.html#scikit-learn-randomforestclassifier-classi
fier
[234]: modules/estimators/classification_scikitlearn.html#scikit-learn-logisticregression-classifier
[235]: modules/estimators/classification_scikitlearn.html#scikit-learn-svc-classifier
[236]: modules/estimators/encoding.html
[237]: modules/estimators/encoding.html#mixin-base-class-encoder
[238]: modules/estimators/encoding.html#tensorflow-encoder
[239]: modules/estimators/gan.html
[240]: modules/estimators/gan.html#tensorflowv2-gan
[241]: modules/estimators/generation.html
[242]: modules/estimators/generation.html#mixin-base-class-generator
[243]: modules/estimators/generation.html#tensorflow-generator
[244]: modules/estimators/generation.html#tensorflow-2-generator
[245]: modules/estimators/object_detection.html
[246]: modules/estimators/object_detection.html#mixin-base-class-object-detector
[247]: modules/estimators/object_detection.html#object-detector-pytorch
[248]: modules/estimators/object_detection.html#object-detector-pytorch-faster-rcnn
[249]: modules/estimators/object_detection.html#object-detector-pytorch-yolo
[250]: modules/estimators/object_detection.html#object-detector-tensorflow-faster-rcnn
[251]: modules/estimators/object_tracking.html
[252]: modules/estimators/object_tracking.html#mixin-base-class-object-tracker
[253]: modules/estimators/object_tracking.html#object-tracker-pytorch-goturn
[254]: modules/estimators/poison_mitigation.html
[255]: modules/estimators/poison_mitigation.html#keras-neural-cleanse-classifier
[256]: modules/estimators/poison_mitigation.html#mixin-base-class-strip
[257]: modules/estimators/regression.html
[258]: modules/estimators/regression.html#mixin-base-class-regressor
[259]: modules/estimators/regression_scikitlearn.html
[260]: modules/estimators/regression_scikitlearn.html#base-class-scikit-learn
[261]: modules/estimators/regression_scikitlearn.html#scikit-learn-decision-tree-regressor
[262]: modules/estimators/speech_recognition.html
[263]: modules/estimators/speech_recognition.html#mixin-base-class-speech-recognizer
[264]: modules/estimators/speech_recognition.html#speech-recognizer-deep-speech-pytorch
[265]: modules/estimators/speech_recognition.html#speech-recognizer-espresso-pytorch
[266]: modules/estimators/speech_recognition.html#speech-recognizer-lingvo-asr-tensorflow
[267]: modules/experimental/estimators.html
[268]: modules/experimental/estimators.html#base-class-jaxestimator
[269]: modules/experimental/estimators/classification.html
[270]: modules/experimental/estimators/classification.html#jax-classifier
[271]: modules/evaluations.html
[272]: modules/metrics.html
[273]: modules/metrics.html#clique-method-robustness-verification
[274]: modules/metrics.html#loss-sensitivity
[275]: modules/metrics.html#empirical-robustness
[276]: modules/metrics.html#clever
[277]: modules/metrics.html#wasserstein-distance
[278]: modules/metrics.html#pointwise-differential-training-privacy
[279]: modules/metrics.html#shapr-membership-privacy-risk
[280]: modules/preprocessing.html
[281]: modules/preprocessing/audio.html
[282]: modules/preprocessing/audio.html#l-filter
[283]: modules/preprocessing/audio.html#lfilter-pytorch
[284]: modules/preprocessing/expectation_over_transformation.html
[285]: modules/preprocessing/expectation_over_transformation.html#eot-image-center-crop-pytorch
[286]: modules/preprocessing/expectation_over_transformation.html#eot-image-rotation-tensorflow-v2
[287]: modules/preprocessing/expectation_over_transformation.html#eot-image-rotation-pytorch
[288]: modules/preprocessing/expectation_over_transformation.html#eot-brightness-pytorch
[289]: modules/preprocessing/expectation_over_transformation.html#eot-brightness-tensorflow-v2
[290]: modules/preprocessing/expectation_over_transformation.html#eot-contrast-pytorch
[291]: modules/preprocessing/expectation_over_transformation.html#eot-contrast-tensorflow-v2
[292]: modules/preprocessing/expectation_over_transformation.html#eot-gaussian-noise-pytorch
[293]: modules/preprocessing/expectation_over_transformation.html#eot-gaussian-noise-tensorflow-v2
[294]: modules/preprocessing/expectation_over_transformation.html#eot-shot-noise-pytorch
[295]: modules/preprocessing/expectation_over_transformation.html#eot-shot-noise-tensorflow-v2
[296]: modules/preprocessing/expectation_over_transformation.html#eot-zoom-blur-pytorch
[297]: modules/preprocessing/expectation_over_transformation.html#eot-zoom-blur-tensorflow-v2
[298]: modules/preprocessing/standardisation_mean_std.html
[299]: modules/preprocessing/standardisation_mean_std.html#standardisation-mean-and-std
[300]: modules/preprocessing/standardisation_mean_std.html#standardisation-mean-and-std-pytorch
[301]: modules/preprocessing/standardisation_mean_std.html#standardisation-mean-and-std-tensorflow-v
2
[302]: modules/data_generators.html
[303]: modules/data_generators.html#base-class
[304]: modules/data_generators.html#framework-specific-data-generators
[305]: modules/exceptions.html
[306]: modules/exceptions.html#estimatorerror
[307]: modules/summary_writer.html
[308]: modules/summary_writer.html#base-class-summarywriter
[309]: modules/summary_writer.html#summary-writer-default
[310]: modules/utils.html
[311]: modules/utils.html#deprecation-operations
[312]: modules/utils.html#math-operations
[313]: modules/utils.html#label-operations
[314]: modules/utils.html#dataset-operations
[315]: modules/tests/utils.html
[316]: modules/tests/utils.html#test-base-classes
[317]: modules/tests/utils.html#trained-models-for-unittests-mnist
[318]: modules/tests/utils.html#trained-models-for-unittests-iris
[319]: modules/tests/utils.html#random-number-generators
[320]: #indices-and-tables
[321]: genindex.html
[322]: py-modindex.html
[323]: search.html
[324]: #
[325]: guide/setup.html
[326]: guide/examples.html
[327]: guide/notebooks.html
[328]: modules/attacks.html
[329]: modules/attacks/evasion.html
[330]: modules/attacks/extraction.html
[331]: modules/attacks/inference/attribute_inference.html
[332]: modules/attacks/inference/membership_inference.html
[333]: modules/attacks/inference/model_inversion.html
[334]: modules/attacks/inference/reconstruction.html
[335]: modules/attacks/poisoning.html
[336]: modules/defences.html
[337]: modules/defences/detector_evasion.html
[338]: modules/defences/detector_poisoning.html
[339]: modules/defences/postprocessor.html
[340]: modules/defences/preprocessor.html
[341]: modules/defences/trainer.html
[342]: modules/defences/transformer_evasion.html
[343]: modules/defences/transformer_poisoning.html
[344]: modules/estimators.html
[345]: modules/estimators/certification.html
[346]: modules/estimators/certification_deep_z.html
[347]: modules/estimators/certification_interval.html
[348]: modules/estimators/certification_randomized_smoothing.html
[349]: modules/estimators/classification.html
[350]: modules/estimators/classification_scikitlearn.html
[351]: modules/estimators/encoding.html
[352]: modules/estimators/gan.html
[353]: modules/estimators/generation.html
[354]: modules/estimators/object_detection.html
[355]: modules/estimators/object_tracking.html
[356]: modules/estimators/poison_mitigation.html
[357]: modules/estimators/regression.html
[358]: modules/estimators/regression_scikitlearn.html
[359]: modules/estimators/speech_recognition.html
[360]: modules/experimental/estimators.html
[361]: modules/experimental/estimators/classification.html
[362]: modules/evaluations.html
[363]: modules/metrics.html
[364]: modules/preprocessing.html
[365]: modules/preprocessing/audio.html
[366]: modules/preprocessing/expectation_over_transformation.html
[367]: modules/preprocessing/standardisation_mean_std.html
[368]: modules/data_generators.html
[369]: modules/exceptions.html
[370]: modules/summary_writer.html
[371]: modules/utils.html
[372]: modules/tests/utils.html
[373]: #
[374]: guide/setup.html
[375]: https://www.sphinx-doc.org/
[376]: https://alabaster.readthedocs.io
[377]: _sources/index.rst.txt
