* TextAttack Documentation
* [ View page source][1]

# TextAttack Documentation[][2]

Get Started

* [Basic-Introduction][3]
  
  * [What is TextAttack?][4]
  * [Where should I start?][5]
  * [NLP Attacks][6]
  * [Data Augmentation][7]
  * [Features][8]
* [Installation][9]
  
  * [Install with pip][10]
  * [Install from Source][11]
  * [Optional Dependencies][12]
  * [FAQ on installation][13]
* [Command-Line Usage][14]
  
  * [Data Augmentation with `textattack augment`][15]
  * [Adversarial Attacks with `textattack attack`][16]
  * [Training Models with `textattack train`][17]
    
    * [Available Models][18]
      
      * [TextAttack Models][19]
      * [`transformers` Models][20]
  * [Evaluating Models with `textattack eval-model`][21]
  * [Other Commands][22]
    
    * [Checkpoints and `textattack attack-resume`][23]
    * [Listing features with `textattack list`][24]
    * [Examining datasets with `textattack peek-dataset`][25]
* [Quick API Usage][26]
  
  * [Attacking a BERT model][27]
* [FAQ][28]
  
  * [Via Slack: Where to Ask Questions:][29]
  * [Via CLI: `--help`][30]
  * [Via our papers: More details on results][31]
  * [Via readthedocs: More details on APIs][32]
  * [More Concrete Questions:][33]
    
    * [0. For many of the dependent library issues, the following command is the first you could
      try:][34]
    * [1. How to Train][35]
    * [2. Use Custom Models][36]
      
      * [Model from a file][37]
    * [3. Use Custom Datasets][38]
      
      * [From a file][39]
      * [Dataset loading via other mechanism, see: more details at here][40]
      * [Custom Dataset via AttackedText class][41]
    * [4. Benchmarking Attacks][42]
    * [5. Create Custom or New Attacks][43]
    * [6. The attacking is too slow][44]

Recipes

* [Attack Recipes CommandLine Use][45]
  
  * [Help: `textattack --help`][46]
  * [Running Attacks: `textattack attack --help`][47]
  * [Attacks and Papers Implemented (”Attack Recipes”): `textattack attack --recipe
    [recipe_name]`][48]
  * [Recipe Usage Examples][49]
* [Attack Recipes API][50]
  
  * [Attacks on classification models][51]
    
    * [A2T (A2T: Attack for Adversarial Training Recipe)][52]
    * [Alzantot Genetic Algorithm][53]
    * [Faster Alzantot Genetic Algorithm][54]
    * [BAE (BAE: BERT-Based Adversarial Examples)][55]
    * [BERT-Attack:][56]
    * [CheckList:][57]
    * [DeepWordBug][58]
    * [HotFlip][59]
    * [Improved Genetic Algorithm][60]
    * [Input Reduction][61]
    * [Kuleshov2017][62]
    * [Particle Swarm Optimization][63]
    * [PWWS][64]
    * [TextFooler (Is BERT Really Robust?)][65]
    * [TextBugger][66]
    * [CLARE Recipe][67]
    * [Pruthi2019: Combating with Robust Word Recognition][68]
  * [Attacks on sequence-to-sequence models][69]
    
    * [MORPHEUS2020][70]
    * [Seq2Sick][71]
  * [General][72]
    
    * [Imperceptible Perturbations Algorithm][73]
* [Augmenter Recipes CommandLine Use][74]
  
  * [Augmenting Text: `textattack augment`][75]
  * [Augmentation Command-Line Interface][76]
* [Augmenter Recipes API][77]
  
  * [Augmenter Recipes:][78]
* [TextAttack Model Zoo][79]
  
  * [Available Models][80]
    
    * [TextAttack Models][81]
    * [`transformers` Models][82]
  * [Evaluation Results of Available Models][83]
    
    * [`LSTM`][84]
    * [`wordCNN`][85]
    * [`albert-base-v2`][86]
    * [`bert-base-uncased`][87]
    * [`distilbert-base-cased`][88]
    * [`distilbert-base-uncased`][89]
    * [`roberta-base`][90]
    * [`xlnet-base-cased`][91]
  * [How we have trained the TextAttack Models][92]
  * [Training details for each TextAttack Model][93]
  * [More details on TextAttack fine-tuned NLP models (details on target NLP task, input type,
    output type, SOTA results on paperswithcode; model card on huggingface):][94]

Using TextAttack

* [What is an adversarial attack in NLP?][95]
  
  * [Terminology][96]
  * [Adversarial Examples in NLP][97]
  * [Generating adversarial examples with TextAttack][98]
  * [The future of adversarial attacks in NLP][99]
* [How to Cite TextAttack][100]
  
  * [Main Paper: TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial
    Training in NLP][101]
    
    * [Our Github on TextAttack: https://github.com/QData/TextAttack][102]
  * [Our Analysis paper: Reevaluating Adversarial Examples in Natural Language][103]
    
    * [Our Github on Reevaluation: Reevaluating-NLP-Adversarial-Examples Github][104]
  * [Our Analysis paper: Searching for a Search Method: Benchmarking Search Algorithms for
    Generating NLP Adversarial Examples][105]
    
    * [Our Github on benchmarking: TextAttack-Search-Benchmark Github][106]
  * [Our defense paper: Title: “Towards Improving Adversarial Training of NLP Models”][107]
    
    * [Code is available][108]
    * [Citations:][109]
  * [Our extended use case paper: “Expanding Scope: Adapting English Adversarial Attacks to
    Chinese”][110]
    
    * [Abstract:][111]
    * [Venue:][112]
    * [Tutorial code:][113]
    * [Citations:][114]
* [Four Components of TextAttack Attacks][115]
  
  * [Goal Functions][116]
  * [Constraints][117]
  * [Transformations][118]
  * [Search Methods][119]
  * [On Benchmarking Attack Recipes][120]
  * [Four components in Attack Recipes we have implemented][121]
* [Benchmarking Search Algorithms for Generating NLP Adversarial Examples][122]
  
  * [Title: Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP
    Adversarial Examples][123]
  * [Our search benchmarking result Github][124]
  * [Our benchmarking results on comparing search methods used in the past attacks.][125]
  * [Benchmarking Attack Recipes][126]
* [On Quality of Generated Adversarial Examples and How to Set Attack Contraints][127]
  
  * [Title: Reevaluating Adversarial Examples in Natural Language][128]
  * [Our Github on Reevaluation: Reevaluating-NLP-Adversarial-Examples Github][129]
  * [Some of our evaluation results on quality of two SOTA attack recipes][130]
  * [Some of our evaluation results on how to set constraints to evaluate NLP model’s adversarial
    robustness][131]
* [Making Vanilla Adversarial Training of NLP Models Feasible!][132]
  
  * [Title: Towards Improving Adversarial Training of NLP Models][133]
    
    * [Video recording of this talk:
      https://underline.io/events/192/sessions/7928/lecture/38377-towards-improving-adversarial-trai
      ning-of-nlp-models][134]
  * [Code is available][135]
  * [Citations:][136]
  * [A2T Attack Recipe][137]
* [Lessons learned in designing TextAttack][138]
  
  * [Presentations on TextAttack][139]
    
    * [2020: Jack Morris’ summary tutorial talk on TextAttack][140]
    * [2021: Dr. Qi’s summary tutorial talk on TextAttack][141]
  * [Challenges in Design][142]
  * [Our design tips][143]
  * [TextAttack flowchart][144]
  * [More Details in Reference][145]
* [TextAttack Extended Functions (Multilingual)][146]
  
  * [Textattack Supports Multiple Model Types besides huggingface models and our textattack
    models:][147]
  * [Multilingual Supports][148]
  * [User defined custom inputs and models][149]
    
    * [Custom Datasets: Dataset from a file][150]
      
      * [Custom Model: from a file][151]
  * [User defined Custom attack components][152]
  * [Visulizing TextAttack generated Examples;][153]
* [How can I contribute to TextAttack?][154]
  
  * [Slack Channel][155]
  * [Ways to contribute][156]
  * [Submitting a new issue or feature request][157]
    
    * [Found a bug?][158]
    * [Do you want to add your model?][159]
    * [Do you want a new feature: a component, a recipe, or something else?][160]
  * [Start contributing! (Pull Requests)][161]
    
    * [Checklist][162]
    * [Tests][163]
      
      * [This guide was heavily inspired by the awesome transformers guide to contributing][164]

Notebook Tutorials

* [Tutorial 0: TextAttack End-To-End (Train, Eval, Attack)][165]
  
  * [Training][166]
  * [Evaluation][167]
  * [Attack][168]
  * [Conclusion][169]
  * [Bonus][170]
* [Tutorial 1: Transformations][171]
  
  * [Goal function][172]
  * [Search method][173]
  * [Transformation][174]
  * [Constraints][175]
  * [A custom transformation][176]
  * [Banana word swap][177]
  * [Using our transformation][178]
  * [Creating the goal function, model, and dataset][179]
  * [Creating the attack][180]
  * [Using the attack][181]
  * [Visualizing attack results][182]
  * [Conclusion][183]
  * [Bonus: Attacking Custom Samples][184]
* [Tutorial 2: Constraints][185]
  
  * [Classes of constraints][186]
  * [A new constraint][187]
  * [A custom constraint][188]
  * [NLTK and Named Entity Recognition][189]
  * [NLTK NER Example][190]
  * [Caching with `@functools.lru_cache`][191]
  * [Putting it all together: getting a list of Named Entity Labels from a sentence][192]
  * [Creating our NamedEntityConstraint][193]
  * [Testing our constraint][194]
  * [Conclusion][195]
* [Tutorial 3: Augmentation][196]
  
  * [Creating an Augmenter][197]
  * [Pre-built Augmentation Recipes][198]
  * [Conclusion][199]
* [Tutorial 4: Custom Word Embeddings][200]
  
  * [**Importing the Model**][201]
  * [**Creating A Custom Dataset**][202]
  * [**Creating An Attack**][203]
  * [**Attack Results With Custom Dataset**][204]
  * [**Creating A Custom Word Embedding**][205]
  * [**Attack Results With Custom Dataset and Word Embedding**][206]
* [Tutorial 5: Attacking TensorFlow models][207]
  
  * [Run textattack on a trained tensorflow model:][208]
    
    * [First: Training][209]
    * [Attacking][210]
  * [Conclusion][211]
* [Tutorial 6: Attacking scikit-learn models][212]
  
  * [Training][213]
  * [Attacking][214]
  * [Conclusion][215]
* [Tutorial 7: Attacking AllenNLP models][216]
* [Tutorial 8: Attacking Keras models][217]
  
  * [This notebook runs textattack on a trained keras model:][218]
  * [Training][219]
  * [Attacking][220]
  * [Conclusion][221]
* [Tutorial 9: Attacking multilingual models][222]
* [Tutorial10: Explaining Attacking BERT model using Captum][223]
* [Tutorial11: Attacking multilingual - Chinese NLP model using Textattack][224]

API User Guide

* [Attack][225]
  
  * [Attack][226]
    
    * [`Attack`][227]
      
      * [`Attack.attack()`][228]
      * [`Attack.cpu_()`][229]
      * [`Attack.cuda_()`][230]
      * [`Attack.filter_transformations()`][231]
      * [`Attack.get_indices_to_order()`][232]
      * [`Attack.get_transformations()`][233]
  * [AttackRecipe][234]
    
    * [`AttackRecipe`][235]
      
      * [`AttackRecipe.build()`][236]
* [Attacker][237]
  
  * [Attacker][238]
    
    * [`Attacker`][239]
      
      * [`Attacker.attack_dataset()`][240]
      * [`Attacker.from_checkpoint()`][241]
      * [`Attacker.update_attack_args()`][242]
  * [AttackArgs][243]
    
    * [`AttackArgs`][244]
      
      * [`AttackArgs.create_loggers_from_args()`][245]
* [AttackResult][246]
  
  * [AttackResult][247]
    
    * [`AttackResult`][248]
      
      * [`AttackResult.diff_color()`][249]
      * [`AttackResult.goal_function_result_str()`][250]
      * [`AttackResult.original_text()`][251]
      * [`AttackResult.perturbed_text()`][252]
      * [`AttackResult.str_lines()`][253]
  * [SuccessfulAttackResult][254]
    
    * [`SuccessfulAttackResult`][255]
  * [FailedAttackResult][256]
    
    * [`FailedAttackResult`][257]
      
      * [`FailedAttackResult.goal_function_result_str()`][258]
      * [`FailedAttackResult.str_lines()`][259]
  * [SkippedAttackResult][260]
    
    * [`SkippedAttackResult`][261]
      
      * [`SkippedAttackResult.goal_function_result_str()`][262]
      * [`SkippedAttackResult.str_lines()`][263]
  * [MaximizedAttackResult][264]
    
    * [`MaximizedAttackResult`][265]
* [Trainer][266]
  
  * [Trainer][267]
    
    * [`Trainer`][268]
      
      * [`Trainer.evaluate()`][269]
      * [`Trainer.evaluate_step()`][270]
      * [`Trainer.get_eval_dataloader()`][271]
      * [`Trainer.get_optimizer_and_scheduler()`][272]
      * [`Trainer.get_train_dataloader()`][273]
      * [`Trainer.train()`][274]
      * [`Trainer.training_step()`][275]
  * [TrainingArgs][276]
    
    * [`TrainingArgs`][277]
* [Datasets][278]
  
  * [Dataset][279]
    
    * [`Dataset`][280]
      
      * [`Dataset.__getitem__()`][281]
      * [`Dataset.__len__()`][282]
  * [HuggingFaceDataset][283]
    
    * [`HuggingFaceDataset`][284]
      
      * [`HuggingFaceDataset.__getitem__()`][285]
      * [`HuggingFaceDataset.__len__()`][286]
* [GoalFunction][287]
  
  * [GoalFunction][288]
    
    * [`GoalFunction`][289]
      
      * [`GoalFunction.extra_repr_keys()`][290]
      * [`GoalFunction.get_output()`][291]
      * [`GoalFunction.get_result()`][292]
      * [`GoalFunction.get_results()`][293]
      * [`GoalFunction.init_attack_example()`][294]
  * [LogitSum][295]
    
    * [`LogitSum`][296]
  * [NamedEntityRecognition][297]
    
    * [`NamedEntityRecognition`][298]
  * [TargetedStrict][299]
    
    * [`TargetedStrict`][300]
  * [TargetedBonus][301]
    
    * [`TargetedBonus`][302]
  * [ClassificationGoalFunction][303]
    
    * [`ClassificationGoalFunction`][304]
      
      * [`ClassificationGoalFunction.extra_repr_keys()`][305]
  * [TargetedClassification][306]
    
    * [`TargetedClassification`][307]
      
      * [`TargetedClassification.extra_repr_keys()`][308]
  * [UntargetedClassification][309]
    
    * [`UntargetedClassification`][310]
  * [InputReduction][311]
    
    * [`InputReduction`][312]
      
      * [`InputReduction.extra_repr_keys()`][313]
  * [TextToTextGoalFunction][314]
    
    * [`TextToTextGoalFunction`][315]
  * [MinimizeBleu][316]
    
    * [`MinimizeBleu`][317]
      
      * [`MinimizeBleu.extra_repr_keys()`][318]
  * [NonOverlappingOutput][319]
    
    * [`NonOverlappingOutput`][320]
  * [MaximizeLevenshtein][321]
    
    * [`MaximizeLevenshtein`][322]
      
      * [`MaximizeLevenshtein.extra_repr_keys()`][323]
* [Constraints][324]
  
  * [Constraint][325]
    
    * [`Constraint`][326]
      
      * [`Constraint.call_many()`][327]
      * [`Constraint.check_compatibility()`][328]
      * [`Constraint.extra_repr_keys()`][329]
  * [PreTransformationConstraint][330]
    
    * [`PreTransformationConstraint`][331]
      
      * [`PreTransformationConstraint.check_compatibility()`][332]
      * [`PreTransformationConstraint.extra_repr_keys()`][333]
* [Transformations][334]
  
  * [Transformation][335]
    
    * [`Transformation`][336]
  * [Composite Transformation][337]
    
    * [`CompositeTransformation`][338]
* [SearchMethod][339]
  
  * [SearchMethod][340]
    
    * [`SearchMethod`][341]
      
      * [`SearchMethod.check_transformation_compatibility()`][342]
      * [`SearchMethod.perform_search()`][343]
      * [`SearchMethod.is_black_box`][344]
  * [BeamSearch][345]
    
    * [`BeamSearch`][346]
      
      * [`BeamSearch.extra_repr_keys()`][347]
      * [`BeamSearch.perform_search()`][348]
      * [`BeamSearch.is_black_box`][349]
  * [GreedySearch][350]
    
    * [`GreedySearch`][351]
      
      * [`GreedySearch.extra_repr_keys()`][352]
  * [GreedyWordSwapWIR][353]
    
    * [`GreedyWordSwapWIR`][354]
      
      * [`GreedyWordSwapWIR.check_transformation_compatibility()`][355]
      * [`GreedyWordSwapWIR.extra_repr_keys()`][356]
      * [`GreedyWordSwapWIR.perform_search()`][357]
      * [`GreedyWordSwapWIR.is_black_box`][358]
  * [AlzantotGeneticAlgorithm][359]
    
    * [`AlzantotGeneticAlgorithm`][360]
  * [ImprovedGeneticAlgorithm][361]
    
    * [`ImprovedGeneticAlgorithm`][362]
      
      * [`ImprovedGeneticAlgorithm.extra_repr_keys()`][363]
  * [ParticleSwarmOptimization][364]
    
    * [`ParticleSwarmOptimization`][365]
      
      * [`ParticleSwarmOptimization.check_transformation_compatibility()`][366]
      * [`ParticleSwarmOptimization.extra_repr_keys()`][367]
      * [`ParticleSwarmOptimization.perform_search()`][368]
      * [`ParticleSwarmOptimization.is_black_box`][369]
  * [DifferentialEvolution][370]
    
    * [`DifferentialEvolution`][371]
      
      * [`DifferentialEvolution.check_transformation_compatibility()`][372]
      * [`DifferentialEvolution.extra_repr_keys()`][373]
      * [`DifferentialEvolution.perform_search()`][374]
      * [`DifferentialEvolution.is_black_box`][375]

Full Reference

* [textattack package][376]
  
  * [Subpackages][377]
    
    * [textattack.attack_recipes package][378]
      
      * [Attack Recipes Package:][379]
      * [Submodules][380]
        
        * [A2T (A2T: Attack for Adversarial Training Recipe)][381]
        * [`A2TYoo2021`][382]
          
          * [`A2TYoo2021.build()`][383]
        * [Attack Recipe Class][384]
        * [`AttackRecipe`][385]
          
          * [`AttackRecipe.build()`][386]
        * [Imperceptible Perturbations Algorithm][387]
        * [`BadCharacters2021`][388]
          
          * [`BadCharacters2021.build()`][389]
        * [BAE (BAE: BERT-Based Adversarial Examples)][390]
        * [`BAEGarg2019`][391]
          
          * [`BAEGarg2019.build()`][392]
        * [BERT-Attack:][393]
        * [`BERTAttackLi2020`][394]
          
          * [`BERTAttackLi2020.build()`][395]
        * [CheckList:][396]
        * [`CheckList2020`][397]
          
          * [`CheckList2020.build()`][398]
        * [Attack Chinese Recipe][399]
        * [`ChineseRecipe`][400]
          
          * [`ChineseRecipe.build()`][401]
        * [CLARE Recipe][402]
        * [`CLARE2020`][403]
          
          * [`CLARE2020.build()`][404]
        * [DeepWordBug][405]
        * [`DeepWordBugGao2018`][406]
          
          * [`DeepWordBugGao2018.build()`][407]
        * [Faster Alzantot Genetic Algorithm][408]
        * [`FasterGeneticAlgorithmJia2019`][409]
          
          * [`FasterGeneticAlgorithmJia2019.build()`][410]
        * [Attack French Recipe][411]
        * [`FrenchRecipe`][412]
          
          * [`FrenchRecipe.build()`][413]
        * [Alzantot Genetic Algorithm][414]
        * [`GeneticAlgorithmAlzantot2018`][415]
          
          * [`GeneticAlgorithmAlzantot2018.build()`][416]
        * [HotFlip][417]
        * [`HotFlipEbrahimi2017`][418]
          
          * [`HotFlipEbrahimi2017.build()`][419]
        * [Improved Genetic Algorithm][420]
        * [`IGAWang2019`][421]
          
          * [`IGAWang2019.build()`][422]
        * [Input Reduction][423]
        * [`InputReductionFeng2018`][424]
          
          * [`InputReductionFeng2018.build()`][425]
        * [Kuleshov2017][426]
        * [`Kuleshov2017`][427]
          
          * [`Kuleshov2017.build()`][428]
        * [MORPHEUS2020][429]
        * [`MorpheusTan2020`][430]
          
          * [`MorpheusTan2020.build()`][431]
        * [Pruthi2019: Combating with Robust Word Recognition][432]
        * [`Pruthi2019`][433]
          
          * [`Pruthi2019.build()`][434]
        * [Particle Swarm Optimization][435]
        * [`PSOZang2020`][436]
          
          * [`PSOZang2020.build()`][437]
        * [PWWS][438]
        * [`PWWSRen2019`][439]
          
          * [`PWWSRen2019.build()`][440]
        * [Seq2Sick][441]
        * [`Seq2SickCheng2018BlackBox`][442]
          
          * [`Seq2SickCheng2018BlackBox.build()`][443]
        * [Attack Spanish Recipe][444]
        * [`SpanishRecipe`][445]
          
          * [`SpanishRecipe.build()`][446]
        * [TextBugger][447]
        * [`TextBuggerLi2018`][448]
          
          * [`TextBuggerLi2018.build()`][449]
        * [TextFooler (Is BERT Really Robust?)][450]
        * [`TextFoolerJin2019`][451]
          
          * [`TextFoolerJin2019.build()`][452]
    * [textattack.attack_results package][453]
      
      * [TextAttack attack_results Package][454]
      * [Submodules][455]
        
        * [AttackResult Class][456]
        * [`AttackResult`][457]
          
          * [`AttackResult.diff_color()`][458]
          * [`AttackResult.goal_function_result_str()`][459]
          * [`AttackResult.original_text()`][460]
          * [`AttackResult.perturbed_text()`][461]
          * [`AttackResult.str_lines()`][462]
        * [FailedAttackResult Class][463]
        * [`FailedAttackResult`][464]
          
          * [`FailedAttackResult.goal_function_result_str()`][465]
          * [`FailedAttackResult.str_lines()`][466]
        * [MaximizedAttackResult Class][467]
        * [`MaximizedAttackResult`][468]
        * [SkippedAttackResult Class][469]
        * [`SkippedAttackResult`][470]
          
          * [`SkippedAttackResult.goal_function_result_str()`][471]
          * [`SkippedAttackResult.str_lines()`][472]
        * [SuccessfulAttackResult Class][473]
        * [`SuccessfulAttackResult`][474]
    * [textattack.augmentation package][475]
      
      * [TextAttack augmentation package:][476]
      * [Submodules][477]
        
        * [Augmenter Class][478]
        * [`AugmentationResult`][479]
          
          * [`AugmentationResult.tempResult`][480]
        * [`Augmenter`][481]
          
          * [`Augmenter.augment()`][482]
          * [`Augmenter.augment_many()`][483]
          * [`Augmenter.augment_text_with_ids()`][484]
        * [Augmenter Recipes:][485]
        * [`BackTranscriptionAugmenter`][486]
        * [`BackTranslationAugmenter`][487]
        * [`CLAREAugmenter`][488]
        * [`CharSwapAugmenter`][489]
        * [`CheckListAugmenter`][490]
        * [`DeletionAugmenter`][491]
        * [`EasyDataAugmenter`][492]
          
          * [`EasyDataAugmenter.augment()`][493]
        * [`EmbeddingAugmenter`][494]
        * [`SwapAugmenter`][495]
        * [`SynonymInsertionAugmenter`][496]
        * [`WordNetAugmenter`][497]
    * [textattack.commands package][498]
      
      * [TextAttack commands Package][499]
      * [Submodules][500]
        
        * [AttackCommand class][501]
        * [`AttackCommand`][502]
          
          * [`AttackCommand.register_subcommand()`][503]
          * [`AttackCommand.run()`][504]
        * [AttackResumeCommand class][505]
        * [`AttackResumeCommand`][506]
          
          * [`AttackResumeCommand.register_subcommand()`][507]
          * [`AttackResumeCommand.run()`][508]
        * [AugmentCommand class][509]
        * [`AugmentCommand`][510]
          
          * [`AugmentCommand.register_subcommand()`][511]
          * [`AugmentCommand.run()`][512]
        * [BenchmarkRecipeCommand class][513]
        * [`BenchmarkRecipeCommand`][514]
          
          * [`BenchmarkRecipeCommand.register_subcommand()`][515]
          * [`BenchmarkRecipeCommand.run()`][516]
        * [EvalModelCommand class][517]
        * [`EvalModelCommand`][518]
          
          * [`EvalModelCommand.get_preds()`][519]
          * [`EvalModelCommand.register_subcommand()`][520]
          * [`EvalModelCommand.run()`][521]
          * [`EvalModelCommand.test_model_on_dataset()`][522]
        * [`ModelEvalArgs`][523]
          
          * [`ModelEvalArgs.batch_size`][524]
          * [`ModelEvalArgs.num_examples`][525]
          * [`ModelEvalArgs.num_examples_offset`][526]
          * [`ModelEvalArgs.random_seed`][527]
        * [ListThingsCommand class][528]
        * [`ListThingsCommand`][529]
          
          * [`ListThingsCommand.register_subcommand()`][530]
          * [`ListThingsCommand.run()`][531]
          * [`ListThingsCommand.things()`][532]
        * [PeekDatasetCommand class][533]
        * [`PeekDatasetCommand`][534]
          
          * [`PeekDatasetCommand.register_subcommand()`][535]
          * [`PeekDatasetCommand.run()`][536]
        * [TextAttack CLI main class][537]
        * [`main()`][538]
        * [`TextAttackCommand`][539]
          
          * [`TextAttackCommand.register_subcommand()`][540]
          * [`TextAttackCommand.run()`][541]
        * [TrainModelCommand class][542]
        * [`TrainModelCommand`][543]
          
          * [`TrainModelCommand.register_subcommand()`][544]
          * [`TrainModelCommand.run()`][545]
    * [textattack.constraints package][546]
      
      * [Constraints][547]
      * [Subpackages][548]
        
        * [textattack.constraints.grammaticality package][549]
          
          * [Grammaticality:][550]
          * [Subpackages][551]
            
            * [textattack.constraints.grammaticality.language_models package][552]
          * [Submodules][553]
            
            * [CoLA for Grammaticality][554]
            * [`COLA`][555]
            * [LanguageTool Grammar Checker][556]
            * [`LanguageTool`][557]
            * [Part of Speech Constraint][558]
            * [`PartOfSpeech`][559]
        * [textattack.constraints.overlap package][560]
          
          * [Overlap Constraints][561]
          * [Submodules][562]
            
            * [BLEU Constraints][563]
            * [`BLEU`][564]
            * [chrF Constraints][565]
            * [`chrF`][566]
            * [Edit Distance Constraints][567]
            * [`LevenshteinEditDistance`][568]
            * [Max Perturb Words Constraints][569]
            * [`MaxWordsPerturbed`][570]
            * [METEOR Constraints][571]
            * [`METEOR`][572]
        * [textattack.constraints.pre_transformation package][573]
          
          * [Pre-Transformation:][574]
          * [Submodules][575]
            
            * [Input Column Modification][576]
            * [`InputColumnModification`][577]
            * [Max Modification Rate][578]
            * [`MaxModificationRate`][579]
            * [Max Modification Rate][580]
            * [`MaxNumWordsModified`][581]
            * [Max Word Index Modification][582]
            * [`MaxWordIndexModification`][583]
            * [Min Word Lenth][584]
            * [`MinWordLength`][585]
            * [Repeat Modification][586]
            * [`RepeatModification`][587]
            * [Stopword Modification][588]
            * [`StopwordModification`][589]
            * [`UnmodifiableIndices`][590]
            * [`UnmodifablePhrases`][591]
        * [textattack.constraints.semantics package][592]
          
          * [Semantic Constraints][593]
          * [Subpackages][594]
            
            * [textattack.constraints.semantics.sentence_encoders package][595]
          * [Submodules][596]
            
            * [BERT Score][597]
            * [`BERTScore`][598]
            * [Word Embedding Distance][599]
            * [`WordEmbeddingDistance`][600]
      * [Submodules][601]
        
        * [TextAttack Constraint Class][602]
        * [`Constraint`][603]
          
          * [`Constraint.call_many()`][604]
          * [`Constraint.check_compatibility()`][605]
          * [`Constraint.extra_repr_keys()`][606]
        * [Pre-Transformation Constraint Class][607]
        * [`PreTransformationConstraint`][608]
          
          * [`PreTransformationConstraint.check_compatibility()`][609]
          * [`PreTransformationConstraint.extra_repr_keys()`][610]
    * [textattack.datasets package][611]
      
      * [datasets package:][612]
      * [Subpackages][613]
        
        * [textattack.datasets.helpers package][614]
          
          * [Dataset Helpers][615]
          * [Submodules][616]
            
            * [Ted Multi TranslationDataset Class][617]
            * [`TedMultiTranslationDataset`][618]
      * [Submodules][619]
        
        * [Dataset Class][620]
        * [`Dataset`][621]
          
          * [`Dataset.filter_by_labels_()`][622]
          * [`Dataset.shuffle()`][623]
        * [HuggingFaceDataset Class][624]
        * [`HuggingFaceDataset`][625]
          
          * [`HuggingFaceDataset.filter_by_labels_()`][626]
          * [`HuggingFaceDataset.shuffle()`][627]
        * [`get_datasets_dataset_columns()`][628]
    * [textattack.goal_function_results package][629]
      
      * [Goal Function Result package:][630]
      * [Subpackages][631]
        
        * [textattack.goal_function_results.custom package][632]
          
          * [Custom Goal Function Result package:][633]
          * [Submodules][634]
            
            * [LogitSumGoalFunctionResult Class][635]
            * [`LogitSumGoalFunctionResult`][636]
            * [NamedEntityRecognitionoalFunctionResult Class][637]
            * [`NamedEntityRecognitionGoalFunctionResult`][638]
            * [TargetedBonusGoalFunctionResult Class][639]
            * [`TargetedBonusGoalFunctionResult`][640]
            * [TargetedStrictGoalFunctionResult Class][641]
            * [`TargetedStrictGoalFunctionResult`][642]
      * [Submodules][643]
        
        * [ClassificationGoalFunctionResult Class][644]
        * [`ClassificationGoalFunctionResult`][645]
          
          * [`ClassificationGoalFunctionResult.get_colored_output()`][646]
          * [`ClassificationGoalFunctionResult.get_text_color_input()`][647]
          * [`ClassificationGoalFunctionResult.get_text_color_perturbed()`][648]
        * [GoalFunctionResult class][649]
        * [`GoalFunctionResult`][650]
          
          * [`GoalFunctionResult.get_colored_output()`][651]
          * [`GoalFunctionResult.get_text_color_input()`][652]
          * [`GoalFunctionResult.get_text_color_perturbed()`][653]
        * [`GoalFunctionResultStatus`][654]
          
          * [`GoalFunctionResultStatus.MAXIMIZING`][655]
          * [`GoalFunctionResultStatus.SEARCHING`][656]
          * [`GoalFunctionResultStatus.SKIPPED`][657]
          * [`GoalFunctionResultStatus.SUCCEEDED`][658]
        * [TextToTextGoalFunctionResult Class][659]
        * [`TextToTextGoalFunctionResult`][660]
          
          * [`TextToTextGoalFunctionResult.get_colored_output()`][661]
          * [`TextToTextGoalFunctionResult.get_text_color_input()`][662]
          * [`TextToTextGoalFunctionResult.get_text_color_perturbed()`][663]
    * [textattack.goal_functions package][664]
      
      * [Goal Functions][665]
      * [Subpackages][666]
        
        * [textattack.goal_functions.classification package][667]
          
          * [Goal fucntion for Classification][668]
          * [Submodules][669]
            
            * [Determine for if an attack has been successful in Classification][670]
            * [`ClassificationGoalFunction`][671]
            * [Determine if an attack has been successful in Hard Label Classficiation.][672]
            * [`HardLabelClassification`][673]
            * [Determine if maintaining the same predicted label (input reduction)][674]
            * [`InputReduction`][675]
            * [Determine if an attack has been successful in targeted Classification][676]
            * [`TargetedClassification`][677]
            * [Determine successful in untargeted Classification][678]
            * [`UntargetedClassification`][679]
        * [textattack.goal_functions.custom package][680]
          
          * [Custom goal functions][681]
          * [Submodules][682]
            
            * [Goal Function for Logit sum][683]
            * [`LogitSum`][684]
            * [Goal Function for NamedEntityRecognition][685]
            * [`NamedEntityRecognition`][686]
            * [Goal Function for Targeted classification with bonus score][687]
            * [`TargetedBonus`][688]
            * [Goal Function for Strict targeted classification][689]
            * [`TargetedStrict`][690]
        * [textattack.goal_functions.text package][691]
          
          * [Goal Function for Text to Text case][692]
          * [Submodules][693]
            
            * [`MaximizeLevenshtein`][694]
            * [Goal Function for Attempts to minimize the BLEU score][695]
            * [`MinimizeBleu`][696]
            * [`get_bleu()`][697]
            * [Goal Function for seq2sick][698]
            * [`NonOverlappingOutput`][699]
            * [`get_words_cached()`][700]
            * [`word_difference_score()`][701]
            * [Goal Function for TextToText][702]
            * [`TextToTextGoalFunction`][703]
      * [Submodules][704]
        
        * [GoalFunction Class][705]
        * [`GoalFunction`][706]
          
          * [`GoalFunction.clear_cache()`][707]
          * [`GoalFunction.extra_repr_keys()`][708]
          * [`GoalFunction.get_output()`][709]
          * [`GoalFunction.get_result()`][710]
          * [`GoalFunction.get_results()`][711]
          * [`GoalFunction.init_attack_example()`][712]
    * [textattack.llms package][713]
      
      * [Large Language Models][714]
      * [Submodules][715]
        
        * [`ChatGptWrapper`][716]
        * [`HuggingFaceLLMWrapper`][717]
    * [textattack.loggers package][718]
      
      * [Misc Loggers: Loggers track, visualize, and export attack results.][719]
      * [Submodules][720]
        
        * [Managing Attack Logs.][721]
        * [`AttackLogManager`][722]
          
          * [`AttackLogManager.add_output_csv()`][723]
          * [`AttackLogManager.add_output_file()`][724]
          * [`AttackLogManager.add_output_summary_json()`][725]
          * [`AttackLogManager.disable_color()`][726]
          * [`AttackLogManager.enable_stdout()`][727]
          * [`AttackLogManager.enable_visdom()`][728]
          * [`AttackLogManager.enable_wandb()`][729]
          * [`AttackLogManager.flush()`][730]
          * [`AttackLogManager.log_attack_details()`][731]
          * [`AttackLogManager.log_result()`][732]
          * [`AttackLogManager.log_results()`][733]
          * [`AttackLogManager.log_sep()`][734]
          * [`AttackLogManager.log_summary()`][735]
          * [`AttackLogManager.log_summary_rows()`][736]
          * [`AttackLogManager.metrics`][737]
        * [Attack Logs to CSV][738]
        * [`CSVLogger`][739]
          
          * [`CSVLogger.close()`][740]
          * [`CSVLogger.flush()`][741]
          * [`CSVLogger.log_attack_result()`][742]
        * [Attack Logs to file][743]
        * [`FileLogger`][744]
          
          * [`FileLogger.close()`][745]
          * [`FileLogger.flush()`][746]
          * [`FileLogger.log_attack_result()`][747]
          * [`FileLogger.log_sep()`][748]
          * [`FileLogger.log_summary_rows()`][749]
        * [Attack Summary Results Logs to Json][750]
        * [`JsonSummaryLogger`][751]
          
          * [`JsonSummaryLogger.close()`][752]
          * [`JsonSummaryLogger.flush()`][753]
          * [`JsonSummaryLogger.log_summary_rows()`][754]
        * [Attack Logger Wrapper][755]
        * [`Logger`][756]
          
          * [`Logger.close()`][757]
          * [`Logger.flush()`][758]
          * [`Logger.log_attack_result()`][759]
          * [`Logger.log_hist()`][760]
          * [`Logger.log_sep()`][761]
          * [`Logger.log_summary_rows()`][762]
        * [Attack Logs to Visdom][763]
        * [`VisdomLogger`][764]
          
          * [`VisdomLogger.bar()`][765]
          * [`VisdomLogger.flush()`][766]
          * [`VisdomLogger.hist()`][767]
          * [`VisdomLogger.log_attack_result()`][768]
          * [`VisdomLogger.log_hist()`][769]
          * [`VisdomLogger.log_summary_rows()`][770]
          * [`VisdomLogger.table()`][771]
          * [`VisdomLogger.text()`][772]
        * [`port_is_open()`][773]
        * [Attack Logs to WandB][774]
        * [`WeightsAndBiasesLogger`][775]
          
          * [`WeightsAndBiasesLogger.log_attack_result()`][776]
          * [`WeightsAndBiasesLogger.log_sep()`][777]
          * [`WeightsAndBiasesLogger.log_summary_rows()`][778]
    * [textattack.metrics package][779]
      
      * [metrics package: to calculate advanced metrics for evaluting attacks and augmented
        text][780]
      * [Subpackages][781]
        
        * [textattack.metrics.attack_metrics package][782]
          
          * [attack_metrics package:][783]
          * [Submodules][784]
            
            * [Metrics on AttackQueries][785]
            * [`AttackQueries`][786]
            * [Metrics on AttackSuccessRate][787]
            * [`AttackSuccessRate`][788]
            * [Metrics on perturbed words][789]
            * [`WordsPerturbed`][790]
        * [textattack.metrics.quality_metrics package][791]
          
          * [Metrics on Quality package][792]
          * [Submodules][793]
            
            * [BERTScoreMetric class:][794]
            * [`BERTScoreMetric`][795]
            * [MeteorMetric class:][796]
            * [`MeteorMetric`][797]
            * [Perplexity Metric:][798]
            * [`Perplexity`][799]
            * [USEMetric class:][800]
            * [`SBERTMetric`][801]
            * [USEMetric class:][802]
            * [`USEMetric`][803]
      * [Submodules][804]
        
        * [Metric Class][805]
        * [`Metric`][806]
          
          * [`Metric.calculate()`][807]
        * [Attack Metric Quality Recipes:][808]
        * [`AdvancedAttackMetric`][809]
          
          * [`AdvancedAttackMetric.calculate()`][810]
    * [textattack.models package][811]
      
      * [Models][812]
        
        * [Models User-specified][813]
        * [Models Pre-trained][814]
        * [Model Wrappers][815]
      * [Subpackages][816]
        
        * [textattack.models.helpers package][817]
          
          * [Moderl Helpers][818]
          * [Submodules][819]
            
            * [Glove Embedding][820]
            * [`EmbeddingLayer`][821]
            * [`GloveEmbeddingLayer`][822]
            * [LSTM 4 Classification][823]
            * [`LSTMForClassification`][824]
            * [T5 model trained to generate text from text][825]
            * [`T5ForTextToText`][826]
            * [Util function for Model Wrapper][827]
            * [`load_cached_state_dict()`][828]
            * [Word CNN for Classification][829]
            * [`CNNTextLayer`][830]
            * [`WordCNNForClassification`][831]
        * [textattack.models.tokenizers package][832]
          
          * [Tokenizers for Model Wrapper][833]
          * [Submodules][834]
            
            * [Glove Tokenizer][835]
            * [`GloveTokenizer`][836]
            * [`WordLevelTokenizer`][837]
            * [T5 Tokenizer][838]
            * [`T5Tokenizer`][839]
        * [textattack.models.wrappers package][840]
          
          * [Model Wrappers Package][841]
          * [Submodules][842]
            
            * [HuggingFace Model Wrapper][843]
            * [`HuggingFaceModelWrapper`][844]
            * [ModelWrapper class][845]
            * [`ModelWrapper`][846]
            * [PyTorch Model Wrapper][847]
            * [`PyTorchModelWrapper`][848]
            * [scikit-learn Model Wrapper][849]
            * [`SklearnModelWrapper`][850]
            * [TensorFlow Model Wrapper][851]
            * [`TensorFlowModelWrapper`][852]
    * [textattack.prompt_augmentation package][853]
      
      * [Prompt Augmentation][854]
      * [Submodules][855]
        
        * [`PromptAugmentationPipeline`][856]
    * [textattack.search_methods package][857]
      
      * [Search Methods][858]
      * [Submodules][859]
        
        * [Reimplementation of search method from Generating Natural Language Adversarial
          Examples][860]
        * [`AlzantotGeneticAlgorithm`][861]
        * [Beam Search][862]
        * [`BeamSearch`][863]
          
          * [`BeamSearch.extra_repr_keys()`][864]
          * [`BeamSearch.perform_search()`][865]
          * [`BeamSearch.is_black_box`][866]
        * [`DifferentialEvolution`][867]
          
          * [`DifferentialEvolution.check_transformation_compatibility()`][868]
          * [`DifferentialEvolution.extra_repr_keys()`][869]
          * [`DifferentialEvolution.perform_search()`][870]
          * [`DifferentialEvolution.is_black_box`][871]
        * [Genetic Algorithm Word Swap][872]
        * [`GeneticAlgorithm`][873]
          
          * [`GeneticAlgorithm.check_transformation_compatibility()`][874]
          * [`GeneticAlgorithm.extra_repr_keys()`][875]
          * [`GeneticAlgorithm.perform_search()`][876]
          * [`GeneticAlgorithm.is_black_box`][877]
        * [Greedy Search][878]
        * [`GreedySearch`][879]
          
          * [`GreedySearch.extra_repr_keys()`][880]
        * [Greedy Word Swap with Word Importance Ranking][881]
        * [`GreedyWordSwapWIR`][882]
          
          * [`GreedyWordSwapWIR.check_transformation_compatibility()`][883]
          * [`GreedyWordSwapWIR.extra_repr_keys()`][884]
          * [`GreedyWordSwapWIR.perform_search()`][885]
          * [`GreedyWordSwapWIR.is_black_box`][886]
        * [Reimplementation of search method from Xiaosen Wang, Hao Jin, Kun He (2019).][887]
        * [`ImprovedGeneticAlgorithm`][888]
          
          * [`ImprovedGeneticAlgorithm.extra_repr_keys()`][889]
        * [Particle Swarm Optimization][890]
        * [`ParticleSwarmOptimization`][891]
          
          * [`ParticleSwarmOptimization.check_transformation_compatibility()`][892]
          * [`ParticleSwarmOptimization.extra_repr_keys()`][893]
          * [`ParticleSwarmOptimization.perform_search()`][894]
          * [`ParticleSwarmOptimization.is_black_box`][895]
        * [`normalize()`][896]
        * [Population based Search abstract class][897]
        * [`PopulationBasedSearch`][898]
        * [`PopulationMember`][899]
          
          * [`PopulationMember.num_words`][900]
          * [`PopulationMember.score`][901]
          * [`PopulationMember.words`][902]
        * [Search Method Abstract Class][903]
        * [`SearchMethod`][904]
          
          * [`SearchMethod.check_transformation_compatibility()`][905]
          * [`SearchMethod.get_victim_model()`][906]
          * [`SearchMethod.perform_search()`][907]
          * [`SearchMethod.is_black_box`][908]
    * [textattack.shared package][909]
      
      * [Shared TextAttack Functions][910]
      * [Subpackages][911]
        
        * [textattack.shared.utils package][912]
          
          * [Submodules][913]
            
            * [`LazyLoader`][914]
            * [`load_module_from_file()`][915]
            * [`download_from_s3()`][916]
            * [`download_from_url()`][917]
            * [`http_get()`][918]
            * [`path_in_cache()`][919]
            * [`s3_url()`][920]
            * [`set_cache_dir()`][921]
            * [`unzip_file()`][922]
            * [`get_textattack_model_num_labels()`][923]
            * [`hashable()`][924]
            * [`html_style_from_dict()`][925]
            * [`html_table_from_rows()`][926]
            * [`load_textattack_model_from_path()`][927]
            * [`set_seed()`][928]
            * [`sigmoid()`][929]
            * [`ANSI_ESCAPE_CODES`][930]
            * [`ReprMixin`][931]
            * [`TextAttackFlairTokenizer`][932]
            * [`add_indent()`][933]
            * [`check_if_punctuations()`][934]
            * [`check_if_subword()`][935]
            * [`color_from_label()`][936]
            * [`color_from_output()`][937]
            * [`color_text()`][938]
            * [`default_class_repr()`][939]
            * [`flair_tag()`][940]
            * [`has_letter()`][941]
            * [`is_one_word()`][942]
            * [`process_label_name()`][943]
            * [`strip_BPE_artifacts()`][944]
            * [`words_from_text()`][945]
            * [`zip_flair_result()`][946]
            * [`zip_stanza_result()`][947]
            * [`batch_model_predict()`][948]
      * [Submodules][949]
        
        * [Attacked Text Class][950]
        * [`AttackedText`][951]
          
          * [`AttackedText.align_with_model_tokens()`][952]
          * [`AttackedText.all_words_diff()`][953]
          * [`AttackedText.convert_from_original_idxs()`][954]
          * [`AttackedText.delete_word_at_index()`][955]
          * [`AttackedText.first_word_diff()`][956]
          * [`AttackedText.first_word_diff_index()`][957]
          * [`AttackedText.free_memory()`][958]
          * [`AttackedText.generate_new_attacked_text()`][959]
          * [`AttackedText.get_deletion_indices()`][960]
          * [`AttackedText.insert_text_after_word_index()`][961]
          * [`AttackedText.insert_text_before_word_index()`][962]
          * [`AttackedText.ith_word_diff()`][963]
          * [`AttackedText.ner_of_word_index()`][964]
          * [`AttackedText.pos_of_word_index()`][965]
          * [`AttackedText.printable_text()`][966]
          * [`AttackedText.replace_word_at_index()`][967]
          * [`AttackedText.replace_words_at_indices()`][968]
          * [`AttackedText.text_after_word_index()`][969]
          * [`AttackedText.text_until_word_index()`][970]
          * [`AttackedText.text_window_around_index()`][971]
          * [`AttackedText.words_diff_num()`][972]
          * [`AttackedText.words_diff_ratio()`][973]
          * [`AttackedText.SPLIT_TOKEN`][974]
          * [`AttackedText.column_labels`][975]
          * [`AttackedText.newly_swapped_words`][976]
          * [`AttackedText.num_words`][977]
          * [`AttackedText.text`][978]
          * [`AttackedText.tokenizer_input`][979]
          * [`AttackedText.words`][980]
          * [`AttackedText.words_per_input`][981]
        * [Misc Checkpoints][982]
        * [`AttackCheckpoint`][983]
          
          * [`AttackCheckpoint.load()`][984]
          * [`AttackCheckpoint.save()`][985]
          * [`AttackCheckpoint.dataset_offset`][986]
          * [`AttackCheckpoint.datetime`][987]
          * [`AttackCheckpoint.num_failed_attacks`][988]
          * [`AttackCheckpoint.num_maximized_attacks`][989]
          * [`AttackCheckpoint.num_remaining_attacks`][990]
          * [`AttackCheckpoint.num_skipped_attacks`][991]
          * [`AttackCheckpoint.num_successful_attacks`][992]
          * [`AttackCheckpoint.results_count`][993]
        * [Shared data fields][994]
        * [Misc Validators][995]
        * [`transformation_consists_of()`][996]
        * [`transformation_consists_of_word_swaps()`][997]
        * [`transformation_consists_of_word_swaps_and_deletions()`][998]
        * [`transformation_consists_of_word_swaps_differential_evolution()`][999]
        * [`validate_model_goal_function_compatibility()`][1000]
        * [`validate_model_gradient_word_swap_compatibility()`][1001]
        * [Shared loads word embeddings and related distances][1002]
        * [`AbstractWordEmbedding`][1003]
          
          * [`AbstractWordEmbedding.get_cos_sim()`][1004]
          * [`AbstractWordEmbedding.get_mse_dist()`][1005]
          * [`AbstractWordEmbedding.index2word()`][1006]
          * [`AbstractWordEmbedding.nearest_neighbours()`][1007]
          * [`AbstractWordEmbedding.word2index()`][1008]
        * [`GensimWordEmbedding`][1009]
          
          * [`GensimWordEmbedding.get_cos_sim()`][1010]
          * [`GensimWordEmbedding.get_mse_dist()`][1011]
          * [`GensimWordEmbedding.index2word()`][1012]
          * [`GensimWordEmbedding.nearest_neighbours()`][1013]
          * [`GensimWordEmbedding.word2index()`][1014]
        * [`WordEmbedding`][1015]
          
          * [`WordEmbedding.counterfitted_GLOVE_embedding()`][1016]
          * [`WordEmbedding.get_cos_sim()`][1017]
          * [`WordEmbedding.get_mse_dist()`][1018]
          * [`WordEmbedding.index2word()`][1019]
          * [`WordEmbedding.nearest_neighbours()`][1020]
          * [`WordEmbedding.word2index()`][1021]
          * [`WordEmbedding.PATH`][1022]
    * [textattack.transformations package][1023]
      
      * [Transformations][1024]
      * [Subpackages][1025]
        
        * [textattack.transformations.sentence_transformations package][1026]
          
          * [sentence_transformations package][1027]
          * [Submodules][1028]
            
            * [BackTranscription class][1029]
            * [`BackTranscription`][1030]
            * [BackTranslation class][1031]
            * [`BackTranslation`][1032]
            * [SentenceTransformation class][1033]
            * [`SentenceTransformation`][1034]
        * [textattack.transformations.word_insertions package][1035]
          
          * [word_insertions package][1036]
          * [Submodules][1037]
            
            * [WordInsertion Class][1038]
            * [`WordInsertion`][1039]
            * [WordInsertionMaskedLM Class][1040]
            * [`WordInsertionMaskedLM`][1041]
            * [WordInsertionRandomSynonym Class][1042]
            * [`WordInsertionRandomSynonym`][1043]
            * [`check_if_one_word()`][1044]
        * [textattack.transformations.word_merges package][1045]
          
          * [word_merges package][1046]
          * [Submodules][1047]
            
            * [Word Merge][1048]
            * [`WordMerge`][1049]
            * [WordMergeMaskedLM class][1050]
            * [`WordMergeMaskedLM`][1051]
            * [`find_merge_index()`][1052]
        * [textattack.transformations.word_swaps package][1053]
          
          * [word_swaps package][1054]
          * [Subpackages][1055]
            
            * [textattack.transformations.word_swaps.chn_transformations package][1056]
          * [Submodules][1057]
            
            * [Word Swap][1058]
            * [`WordSwap`][1059]
            * [Word Swap by Changing Location][1060]
            * [`WordSwapChangeLocation`][1061]
            * [`idx_to_words()`][1062]
            * [Word Swap by Changing Name][1063]
            * [`WordSwapChangeName`][1064]
            * [Word Swap by Changing Number][1065]
            * [`WordSwapChangeNumber`][1066]
            * [`idx_to_words()`][1067]
            * [Word Swap by Contraction][1068]
            * [`WordSwapContract`][1069]
            * [Word Swap by Invisible Deletions][1070]
            * [`WordSwapDeletions`][1071]
            * [Word Swap for Differential Evolution][1072]
            * [`WordSwapDifferentialEvolution`][1073]
            * [Word Swap by Embedding][1074]
            * [`WordSwapEmbedding`][1075]
            * [`recover_word_case()`][1076]
            * [Word Swap by Extension][1077]
            * [`WordSwapExtend`][1078]
            * [Word Swap by Gradient][1079]
            * [`WordSwapGradientBased`][1080]
            * [Word Swap by Homoglyph][1081]
            * [`WordSwapHomoglyphSwap`][1082]
            * [Word Swap by OpenHowNet][1083]
            * [`WordSwapHowNet`][1084]
            * [`recover_word_case()`][1085]
            * [Word Swap by inflections][1086]
            * [`WordSwapInflections`][1087]
            * [Word Swap by Invisible Characters][1088]
            * [`WordSwapInvisibleCharacters`][1089]
            * [Word Swap by BERT-Masked LM.][1090]
            * [`WordSwapMaskedLM`][1091]
            * [`recover_word_case()`][1092]
            * [Word Swap by Neighboring Character Swap][1093]
            * [`WordSwapNeighboringCharacterSwap`][1094]
            * [Word Swap by swaps characters with QWERTY adjacent keys][1095]
            * [`WordSwapQWERTY`][1096]
            * [Word Swap by Random Character Deletion][1097]
            * [`WordSwapRandomCharacterDeletion`][1098]
            * [Word Swap by Random Character Insertion][1099]
            * [`WordSwapRandomCharacterInsertion`][1100]
            * [Word Swap by Random Character Substitution][1101]
            * [`WordSwapRandomCharacterSubstitution`][1102]
            * [Word Swap by Invisible Reorderings][1103]
            * [`WordSwapReorderings`][1104]
            * [Word Swap by swapping synonyms in WordNet][1105]
            * [`WordSwapWordNet`][1106]
      * [Submodules][1107]
        
        * [Composite Transformation][1108]
        * [`CompositeTransformation`][1109]
        * [Transformation Abstract Class][1110]
        * [`Transformation`][1111]
          
          * [`Transformation.deterministic`][1112]
        * [word deletion Transformation][1113]
        * [`WordDeletion`][1114]
        * [Word Swap Transformation by swapping the order of words][1115]
        * [`WordInnerSwapRandom`][1116]
          
          * [`WordInnerSwapRandom.deterministic`][1117]
  * [Submodules][1118]
    
    * [Attack Class][1119]
    * [`Attack`][1120]
      
      * [`Attack.attack()`][1121]
      * [`Attack.clear_cache()`][1122]
      * [`Attack.cpu_()`][1123]
      * [`Attack.cuda_()`][1124]
      * [`Attack.filter_transformations()`][1125]
      * [`Attack.get_indices_to_order()`][1126]
      * [`Attack.get_transformations()`][1127]
    * [AttackArgs Class][1128]
    * [`AttackArgs`][1129]
      
      * [`AttackArgs.create_loggers_from_args()`][1130]
      * [`AttackArgs.attack_n`][1131]
      * [`AttackArgs.checkpoint_dir`][1132]
      * [`AttackArgs.checkpoint_interval`][1133]
      * [`AttackArgs.csv_coloring_style`][1134]
      * [`AttackArgs.disable_stdout`][1135]
      * [`AttackArgs.enable_advance_metrics`][1136]
      * [`AttackArgs.log_summary_to_json`][1137]
      * [`AttackArgs.log_to_csv`][1138]
      * [`AttackArgs.log_to_txt`][1139]
      * [`AttackArgs.log_to_visdom`][1140]
      * [`AttackArgs.log_to_wandb`][1141]
      * [`AttackArgs.metrics`][1142]
      * [`AttackArgs.num_examples`][1143]
      * [`AttackArgs.num_examples_offset`][1144]
      * [`AttackArgs.num_successful_examples`][1145]
      * [`AttackArgs.num_workers_per_device`][1146]
      * [`AttackArgs.parallel`][1147]
      * [`AttackArgs.query_budget`][1148]
      * [`AttackArgs.random_seed`][1149]
      * [`AttackArgs.shuffle`][1150]
      * [`AttackArgs.silent`][1151]
    * [`CommandLineAttackArgs`][1152]
    * [Attacker Class][1153]
    * [`Attacker`][1154]
      
      * [`Attacker.attack_dataset()`][1155]
      * [`Attacker.attack_interactive()`][1156]
      * [`Attacker.from_checkpoint()`][1157]
      * [`Attacker.update_attack_args()`][1158]
    * [`attack_from_queue()`][1159]
    * [`pytorch_multiprocessing_workaround()`][1160]
    * [`set_env_variables()`][1161]
    * [AugmenterArgs Class][1162]
    * [`AugmenterArgs`][1163]
      
      * [`AugmenterArgs.enable_advanced_metrics`][1164]
      * [`AugmenterArgs.exclude_original`][1165]
      * [`AugmenterArgs.fast_augment`][1166]
      * [`AugmenterArgs.high_yield`][1167]
      * [`AugmenterArgs.input_column`][1168]
      * [`AugmenterArgs.input_csv`][1169]
      * [`AugmenterArgs.interactive`][1170]
      * [`AugmenterArgs.output_csv`][1171]
      * [`AugmenterArgs.overwrite`][1172]
      * [`AugmenterArgs.pct_words_to_swap`][1173]
      * [`AugmenterArgs.random_seed`][1174]
      * [`AugmenterArgs.recipe`][1175]
      * [`AugmenterArgs.transformations_per_example`][1176]
    * [DatasetArgs Class][1177]
    * [`DatasetArgs`][1178]
      
      * [`DatasetArgs.dataset_by_model`][1179]
      * [`DatasetArgs.dataset_from_file`][1180]
      * [`DatasetArgs.dataset_from_huggingface`][1181]
      * [`DatasetArgs.dataset_split`][1182]
      * [`DatasetArgs.filter_by_labels`][1183]
    * [ModelArgs Class][1184]
    * [`ModelArgs`][1185]
      
      * [`ModelArgs.model`][1186]
      * [`ModelArgs.model_from_file`][1187]
      * [`ModelArgs.model_from_huggingface`][1188]
    * [Trainer Class][1189]
    * [`Trainer`][1190]
      
      * [`Trainer.evaluate()`][1191]
      * [`Trainer.evaluate_step()`][1192]
      * [`Trainer.get_eval_dataloader()`][1193]
      * [`Trainer.get_optimizer_and_scheduler()`][1194]
      * [`Trainer.get_train_dataloader()`][1195]
      * [`Trainer.train()`][1196]
      * [`Trainer.training_step()`][1197]
    * [TrainingArgs Class][1198]
    * [`CommandLineTrainingArgs`][1199]
      
      * [`CommandLineTrainingArgs.output_dir`][1200]
    * [`TrainingArgs`][1201]
      
      * [`TrainingArgs.alpha`][1202]
      * [`TrainingArgs.attack_epoch_interval`][1203]
      * [`TrainingArgs.attack_num_workers_per_device`][1204]
      * [`TrainingArgs.checkpoint_interval_epochs`][1205]
      * [`TrainingArgs.checkpoint_interval_steps`][1206]
      * [`TrainingArgs.early_stopping_epochs`][1207]
      * [`TrainingArgs.gradient_accumulation_steps`][1208]
      * [`TrainingArgs.learning_rate`][1209]
      * [`TrainingArgs.load_best_model_at_end`][1210]
      * [`TrainingArgs.log_to_tb`][1211]
      * [`TrainingArgs.log_to_wandb`][1212]
      * [`TrainingArgs.logging_interval_step`][1213]
      * [`TrainingArgs.num_clean_epochs`][1214]
      * [`TrainingArgs.num_epochs`][1215]
      * [`TrainingArgs.num_train_adv_examples`][1216]
      * [`TrainingArgs.num_warmup_steps`][1217]
      * [`TrainingArgs.output_dir`][1218]
      * [`TrainingArgs.parallel`][1219]
      * [`TrainingArgs.per_device_eval_batch_size`][1220]
      * [`TrainingArgs.per_device_train_batch_size`][1221]
      * [`TrainingArgs.query_budget_train`][1222]
      * [`TrainingArgs.random_seed`][1223]
      * [`TrainingArgs.save_last`][1224]
      * [`TrainingArgs.tb_log_dir`][1225]
      * [`TrainingArgs.wandb_project`][1226]
      * [`TrainingArgs.weight_decay`][1227]
    * [`default_output_dir()`][1228]
[Next ][1229]

© Copyright 2021-24, UVA QData Lab.

Built with [Sphinx][1230] using a [theme][1231] provided by [Read the Docs][1232].

[1]: _sources/index.rst.txt
[2]: #textattack-documentation
[3]: 0_get_started/basic-Intro.html
[4]: 0_get_started/basic-Intro.html#what-is-textattack
[5]: 0_get_started/basic-Intro.html#where-should-i-start
[6]: 0_get_started/basic-Intro.html#nlp-attacks
[7]: 0_get_started/basic-Intro.html#data-augmentation
[8]: 0_get_started/basic-Intro.html#features
[9]: 0_get_started/installation.html
[10]: 0_get_started/installation.html#install-with-pip
[11]: 0_get_started/installation.html#install-from-source
[12]: 0_get_started/installation.html#optional-dependencies
[13]: 0_get_started/installation.html#faq-on-installation
[14]: 0_get_started/command_line_usage.html
[15]: 0_get_started/command_line_usage.html#data-augmentation-with-textattack-augment
[16]: 0_get_started/command_line_usage.html#adversarial-attacks-with-textattack-attack
[17]: 0_get_started/command_line_usage.html#training-models-with-textattack-train
[18]: 0_get_started/command_line_usage.html#available-models
[19]: 0_get_started/command_line_usage.html#textattack-models
[20]: 0_get_started/command_line_usage.html#transformers-models
[21]: 0_get_started/command_line_usage.html#evaluating-models-with-textattack-eval-model
[22]: 0_get_started/command_line_usage.html#other-commands
[23]: 0_get_started/command_line_usage.html#checkpoints-and-textattack-attack-resume
[24]: 0_get_started/command_line_usage.html#listing-features-with-textattack-list
[25]: 0_get_started/command_line_usage.html#examining-datasets-with-textattack-peek-dataset
[26]: 0_get_started/quick_api_tour.html
[27]: 0_get_started/quick_api_tour.html#attacking-a-bert-model
[28]: 1start/FAQ.html
[29]: 1start/FAQ.html#via-slack-where-to-ask-questions
[30]: 1start/FAQ.html#via-cli-help
[31]: 1start/FAQ.html#via-our-papers-more-details-on-results
[32]: 1start/FAQ.html#via-readthedocs-more-details-on-apis
[33]: 1start/FAQ.html#more-concrete-questions
[34]: 1start/FAQ.html#for-many-of-the-dependent-library-issues-the-following-command-is-the-first-yo
u-could-try
[35]: 1start/FAQ.html#how-to-train
[36]: 1start/FAQ.html#use-custom-models
[37]: 1start/FAQ.html#model-from-a-file
[38]: 1start/FAQ.html#use-custom-datasets
[39]: 1start/FAQ.html#from-a-file
[40]: 1start/FAQ.html#dataset-loading-via-other-mechanism-see-more-details-at-here
[41]: 1start/FAQ.html#custom-dataset-via-attackedtext-class
[42]: 1start/FAQ.html#benchmarking-attacks
[43]: 1start/FAQ.html#create-custom-or-new-attacks
[44]: 1start/FAQ.html#the-attacking-is-too-slow
[45]: 3recipes/attack_recipes_cmd.html
[46]: 3recipes/attack_recipes_cmd.html#help-textattack-help
[47]: 3recipes/attack_recipes_cmd.html#running-attacks-textattack-attack-help
[48]: 3recipes/attack_recipes_cmd.html#attacks-and-papers-implemented-attack-recipes-textattack-atta
ck-recipe-recipe-name
[49]: 3recipes/attack_recipes_cmd.html#recipe-usage-examples
[50]: 3recipes/attack_recipes.html
[51]: 3recipes/attack_recipes.html#attacks-on-classification-models
[52]: 3recipes/attack_recipes.html#a2t-a2t-attack-for-adversarial-training-recipe
[53]: 3recipes/attack_recipes.html#alzantot-genetic-algorithm
[54]: 3recipes/attack_recipes.html#faster-alzantot-genetic-algorithm
[55]: 3recipes/attack_recipes.html#bae-bae-bert-based-adversarial-examples
[56]: 3recipes/attack_recipes.html#bert-attack
[57]: 3recipes/attack_recipes.html#checklist
[58]: 3recipes/attack_recipes.html#deepwordbug
[59]: 3recipes/attack_recipes.html#hotflip
[60]: 3recipes/attack_recipes.html#improved-genetic-algorithm
[61]: 3recipes/attack_recipes.html#input-reduction
[62]: 3recipes/attack_recipes.html#kuleshov2017
[63]: 3recipes/attack_recipes.html#particle-swarm-optimization
[64]: 3recipes/attack_recipes.html#pwws
[65]: 3recipes/attack_recipes.html#textfooler-is-bert-really-robust
[66]: 3recipes/attack_recipes.html#textbugger
[67]: 3recipes/attack_recipes.html#clare-recipe
[68]: 3recipes/attack_recipes.html#pruthi2019-combating-with-robust-word-recognition
[69]: 3recipes/attack_recipes.html#attacks-on-sequence-to-sequence-models
[70]: 3recipes/attack_recipes.html#morpheus2020
[71]: 3recipes/attack_recipes.html#seq2sick
[72]: 3recipes/attack_recipes.html#general
[73]: 3recipes/attack_recipes.html#imperceptible-perturbations-algorithm
[74]: 3recipes/augmenter_recipes_cmd.html
[75]: 3recipes/augmenter_recipes_cmd.html#augmenting-text-textattack-augment
[76]: 3recipes/augmenter_recipes_cmd.html#augmentation-command-line-interface
[77]: 3recipes/augmenter_recipes.html
[78]: 3recipes/augmenter_recipes.html#augmenter-recipes
[79]: 3recipes/models.html
[80]: 3recipes/models.html#available-models
[81]: 3recipes/models.html#textattack-models
[82]: 3recipes/models.html#transformers-models
[83]: 3recipes/models.html#evaluation-results-of-available-models
[84]: 3recipes/models.html#lstm
[85]: 3recipes/models.html#wordcnn
[86]: 3recipes/models.html#albert-base-v2
[87]: 3recipes/models.html#bert-base-uncased
[88]: 3recipes/models.html#distilbert-base-cased
[89]: 3recipes/models.html#distilbert-base-uncased
[90]: 3recipes/models.html#roberta-base
[91]: 3recipes/models.html#xlnet-base-cased
[92]: 3recipes/models.html#how-we-have-trained-the-textattack-models
[93]: 3recipes/models.html#training-details-for-each-textattack-model
[94]: 3recipes/models.html#more-details-on-textattack-fine-tuned-nlp-models-details-on-target-nlp-ta
sk-input-type-output-type-sota-results-on-paperswithcode-model-card-on-huggingface
[95]: 1start/what_is_an_adversarial_attack.html
[96]: 1start/what_is_an_adversarial_attack.html#terminology
[97]: 1start/what_is_an_adversarial_attack.html#adversarial-examples-in-nlp
[98]: 1start/what_is_an_adversarial_attack.html#generating-adversarial-examples-with-textattack
[99]: 1start/what_is_an_adversarial_attack.html#the-future-of-adversarial-attacks-in-nlp
[100]: 1start/references.html
[101]: 1start/references.html#main-paper-textattack-a-framework-for-adversarial-attacks-data-augment
ation-and-adversarial-training-in-nlp
[102]: 1start/references.html#our-github-on-textattack-https-github-com-qdata-textattack
[103]: 1start/references.html#our-analysis-paper-reevaluating-adversarial-examples-in-natural-langua
ge
[104]: 1start/references.html#our-github-on-reevaluation-reevaluating-nlp-adversarial-examples-githu
b
[105]: 1start/references.html#our-analysis-paper-searching-for-a-search-method-benchmarking-search-a
lgorithms-for-generating-nlp-adversarial-examples
[106]: 1start/references.html#our-github-on-benchmarking-textattack-search-benchmark-github
[107]: 1start/references.html#our-defense-paper-title-towards-improving-adversarial-training-of-nlp-
models
[108]: 1start/references.html#code-is-available
[109]: 1start/references.html#citations
[110]: 1start/references.html#our-extended-use-case-paper-expanding-scope-adapting-english-adversari
al-attacks-to-chinese
[111]: 1start/references.html#abstract
[112]: 1start/references.html#venue
[113]: 1start/references.html#tutorial-code
[114]: 1start/references.html#id1
[115]: 1start/attacks4Components.html
[116]: 1start/attacks4Components.html#goal-functions
[117]: 1start/attacks4Components.html#constraints
[118]: 1start/attacks4Components.html#transformations
[119]: 1start/attacks4Components.html#search-methods
[120]: 1start/attacks4Components.html#on-benchmarking-attack-recipes
[121]: 1start/attacks4Components.html#four-components-in-attack-recipes-we-have-implemented
[122]: 1start/benchmark-search.html
[123]: 1start/benchmark-search.html#title-searching-for-a-search-method-benchmarking-search-algorith
ms-for-generating-nlp-adversarial-examples
[124]: 1start/benchmark-search.html#our-search-benchmarking-result-github
[125]: 1start/benchmark-search.html#our-benchmarking-results-on-comparing-search-methods-used-in-the
-past-attacks
[126]: 1start/benchmark-search.html#benchmarking-attack-recipes
[127]: 1start/quality-SOTA-recipes.html
[128]: 1start/quality-SOTA-recipes.html#title-reevaluating-adversarial-examples-in-natural-language
[129]: 1start/quality-SOTA-recipes.html#our-github-on-reevaluation-reevaluating-nlp-adversarial-exam
ples-github
[130]: 1start/quality-SOTA-recipes.html#some-of-our-evaluation-results-on-quality-of-two-sota-attack
-recipes
[131]: 1start/quality-SOTA-recipes.html#some-of-our-evaluation-results-on-how-to-set-constraints-to-
evaluate-nlp-model-s-adversarial-robustness
[132]: 1start/A2TforVanillaAT.html
[133]: 1start/A2TforVanillaAT.html#title-towards-improving-adversarial-training-of-nlp-models
[134]: 1start/A2TforVanillaAT.html#video-recording-of-this-talk-https-underline-io-events-192-sessio
ns-7928-lecture-38377-towards-improving-adversarial-training-of-nlp-models
[135]: 1start/A2TforVanillaAT.html#code-is-available
[136]: 1start/A2TforVanillaAT.html#citations
[137]: 1start/A2TforVanillaAT.html#a2t-attack-recipe
[138]: 1start/api-design-tips.html
[139]: 1start/api-design-tips.html#presentations-on-textattack
[140]: 1start/api-design-tips.html#jack-morris-summary-tutorial-talk-on-textattack
[141]: 1start/api-design-tips.html#dr-qi-s-summary-tutorial-talk-on-textattack
[142]: 1start/api-design-tips.html#challenges-in-design
[143]: 1start/api-design-tips.html#our-design-tips
[144]: 1start/api-design-tips.html#textattack-flowchart
[145]: 1start/api-design-tips.html#more-details-in-reference
[146]: 1start/multilingual-visualization.html
[147]: 1start/multilingual-visualization.html#textattack-supports-multiple-model-types-besides-huggi
ngface-models-and-our-textattack-models
[148]: 1start/multilingual-visualization.html#multilingual-supports
[149]: 1start/multilingual-visualization.html#user-defined-custom-inputs-and-models
[150]: 1start/multilingual-visualization.html#custom-datasets-dataset-from-a-file
[151]: 1start/multilingual-visualization.html#custom-model-from-a-file
[152]: 1start/multilingual-visualization.html#user-defined-custom-attack-components
[153]: 1start/multilingual-visualization.html#visulizing-textattack-generated-examples
[154]: 1start/support.html
[155]: 1start/support.html#slack-channel
[156]: 1start/support.html#ways-to-contribute
[157]: 1start/support.html#submitting-a-new-issue-or-feature-request
[158]: 1start/support.html#found-a-bug
[159]: 1start/support.html#do-you-want-to-add-your-model
[160]: 1start/support.html#do-you-want-a-new-feature-a-component-a-recipe-or-something-else
[161]: 1start/support.html#start-contributing-pull-requests
[162]: 1start/support.html#checklist
[163]: 1start/support.html#tests
[164]: 1start/support.html#this-guide-was-heavily-inspired-by-the-awesome-transformers-guide-to-cont
ributing
[165]: 2notebook/0_End_to_End.html
[166]: 2notebook/0_End_to_End.html#Training
[167]: 2notebook/0_End_to_End.html#Evaluation
[168]: 2notebook/0_End_to_End.html#Attack
[169]: 2notebook/0_End_to_End.html#Conclusion
[170]: 2notebook/0_End_to_End.html#Bonus
[171]: 2notebook/1_Introduction_and_Transformations.html
[172]: 2notebook/1_Introduction_and_Transformations.html#Goal-function
[173]: 2notebook/1_Introduction_and_Transformations.html#Search-method
[174]: 2notebook/1_Introduction_and_Transformations.html#Transformation
[175]: 2notebook/1_Introduction_and_Transformations.html#Constraints
[176]: 2notebook/1_Introduction_and_Transformations.html#A-custom-transformation
[177]: 2notebook/1_Introduction_and_Transformations.html#Banana-word-swap
[178]: 2notebook/1_Introduction_and_Transformations.html#Using-our-transformation
[179]: 2notebook/1_Introduction_and_Transformations.html#Creating-the-goal-function,-model,-and-data
set
[180]: 2notebook/1_Introduction_and_Transformations.html#Creating-the-attack
[181]: 2notebook/1_Introduction_and_Transformations.html#Using-the-attack
[182]: 2notebook/1_Introduction_and_Transformations.html#Visualizing-attack-results
[183]: 2notebook/1_Introduction_and_Transformations.html#Conclusion
[184]: 2notebook/1_Introduction_and_Transformations.html#Bonus:-Attacking-Custom-Samples
[185]: 2notebook/2_Constraints.html
[186]: 2notebook/2_Constraints.html#Classes-of-constraints
[187]: 2notebook/2_Constraints.html#A-new-constraint
[188]: 2notebook/2_Constraints.html#A-custom-constraint
[189]: 2notebook/2_Constraints.html#NLTK-and-Named-Entity-Recognition
[190]: 2notebook/2_Constraints.html#NLTK-NER-Example
[191]: 2notebook/2_Constraints.html#Caching-with-@functools.lru_cache
[192]: 2notebook/2_Constraints.html#Putting-it-all-together:-getting-a-list-of-Named-Entity-Labels-f
rom-a-sentence
[193]: 2notebook/2_Constraints.html#Creating-our-NamedEntityConstraint
[194]: 2notebook/2_Constraints.html#Testing-our-constraint
[195]: 2notebook/2_Constraints.html#Conclusion
[196]: 2notebook/3_Augmentations.html
[197]: 2notebook/3_Augmentations.html#Creating-an-Augmenter
[198]: 2notebook/3_Augmentations.html#Pre-built-Augmentation-Recipes
[199]: 2notebook/3_Augmentations.html#Conclusion
[200]: 2notebook/4_Custom_Datasets_Word_Embedding.html
[201]: 2notebook/4_Custom_Datasets_Word_Embedding.html#Importing-the-Model
[202]: 2notebook/4_Custom_Datasets_Word_Embedding.html#Creating-A-Custom-Dataset
[203]: 2notebook/4_Custom_Datasets_Word_Embedding.html#Creating-An-Attack
[204]: 2notebook/4_Custom_Datasets_Word_Embedding.html#Attack-Results-With-Custom-Dataset
[205]: 2notebook/4_Custom_Datasets_Word_Embedding.html#Creating-A-Custom-Word-Embedding
[206]: 2notebook/4_Custom_Datasets_Word_Embedding.html#Attack-Results-With-Custom-Dataset-and-Word-E
mbedding
[207]: 2notebook/Example_0_tensorflow.html
[208]: 2notebook/Example_0_tensorflow.html#Run-textattack-on-a-trained-tensorflow-model:
[209]: 2notebook/Example_0_tensorflow.html#First:-Training
[210]: 2notebook/Example_0_tensorflow.html#Attacking
[211]: 2notebook/Example_0_tensorflow.html#Conclusion
[212]: 2notebook/Example_1_sklearn.html
[213]: 2notebook/Example_1_sklearn.html#Training
[214]: 2notebook/Example_1_sklearn.html#Attacking
[215]: 2notebook/Example_1_sklearn.html#Conclusion
[216]: 2notebook/Example_2_allennlp.html
[217]: 2notebook/Example_3_Keras.html
[218]: 2notebook/Example_3_Keras.html#This-notebook-runs-textattack-on-a-trained-keras-model:
[219]: 2notebook/Example_3_Keras.html#Training
[220]: 2notebook/Example_3_Keras.html#Attacking
[221]: 2notebook/Example_3_Keras.html#Conclusion
[222]: 2notebook/Example_4_CamemBERT.html
[223]: 2notebook/Example_5_Explain_BERT.html
[224]: 2notebook/Example_6_Chinese_Attack.html
[225]: api/attack.html
[226]: api/attack.html#attack
[227]: api/attack.html#textattack.Attack
[228]: api/attack.html#textattack.Attack.attack
[229]: api/attack.html#textattack.Attack.cpu_
[230]: api/attack.html#textattack.Attack.cuda_
[231]: api/attack.html#textattack.Attack.filter_transformations
[232]: api/attack.html#textattack.Attack.get_indices_to_order
[233]: api/attack.html#textattack.Attack.get_transformations
[234]: api/attack.html#attackrecipe
[235]: api/attack.html#textattack.attack_recipes.AttackRecipe
[236]: api/attack.html#textattack.attack_recipes.AttackRecipe.build
[237]: api/attacker.html
[238]: api/attacker.html#attacker
[239]: api/attacker.html#textattack.Attacker
[240]: api/attacker.html#textattack.Attacker.attack_dataset
[241]: api/attacker.html#textattack.Attacker.from_checkpoint
[242]: api/attacker.html#textattack.Attacker.update_attack_args
[243]: api/attacker.html#attackargs
[244]: api/attacker.html#textattack.AttackArgs
[245]: api/attacker.html#textattack.AttackArgs.create_loggers_from_args
[246]: api/attack_results.html
[247]: api/attack_results.html#attackresult
[248]: api/attack_results.html#textattack.attack_results.AttackResult
[249]: api/attack_results.html#textattack.attack_results.AttackResult.diff_color
[250]: api/attack_results.html#textattack.attack_results.AttackResult.goal_function_result_str
[251]: api/attack_results.html#textattack.attack_results.AttackResult.original_text
[252]: api/attack_results.html#textattack.attack_results.AttackResult.perturbed_text
[253]: api/attack_results.html#textattack.attack_results.AttackResult.str_lines
[254]: api/attack_results.html#successfulattackresult
[255]: api/attack_results.html#textattack.attack_results.SuccessfulAttackResult
[256]: api/attack_results.html#failedattackresult
[257]: api/attack_results.html#textattack.attack_results.FailedAttackResult
[258]: api/attack_results.html#textattack.attack_results.FailedAttackResult.goal_function_result_str
[259]: api/attack_results.html#textattack.attack_results.FailedAttackResult.str_lines
[260]: api/attack_results.html#skippedattackresult
[261]: api/attack_results.html#textattack.attack_results.SkippedAttackResult
[262]: api/attack_results.html#textattack.attack_results.SkippedAttackResult.goal_function_result_st
r
[263]: api/attack_results.html#textattack.attack_results.SkippedAttackResult.str_lines
[264]: api/attack_results.html#maximizedattackresult
[265]: api/attack_results.html#textattack.attack_results.MaximizedAttackResult
[266]: api/trainer.html
[267]: api/trainer.html#trainer
[268]: api/trainer.html#textattack.Trainer
[269]: api/trainer.html#textattack.Trainer.evaluate
[270]: api/trainer.html#textattack.Trainer.evaluate_step
[271]: api/trainer.html#textattack.Trainer.get_eval_dataloader
[272]: api/trainer.html#textattack.Trainer.get_optimizer_and_scheduler
[273]: api/trainer.html#textattack.Trainer.get_train_dataloader
[274]: api/trainer.html#textattack.Trainer.train
[275]: api/trainer.html#textattack.Trainer.training_step
[276]: api/trainer.html#trainingargs
[277]: api/trainer.html#textattack.TrainingArgs
[278]: api/datasets.html
[279]: api/datasets.html#dataset
[280]: api/datasets.html#textattack.datasets.Dataset
[281]: api/datasets.html#textattack.datasets.Dataset.__getitem__
[282]: api/datasets.html#textattack.datasets.Dataset.__len__
[283]: api/datasets.html#huggingfacedataset
[284]: api/datasets.html#textattack.datasets.HuggingFaceDataset
[285]: api/datasets.html#textattack.datasets.HuggingFaceDataset.__getitem__
[286]: api/datasets.html#textattack.datasets.HuggingFaceDataset.__len__
[287]: api/goal_functions.html
[288]: api/goal_functions.html#goalfunction
[289]: api/goal_functions.html#textattack.goal_functions.GoalFunction
[290]: api/goal_functions.html#textattack.goal_functions.GoalFunction.extra_repr_keys
[291]: api/goal_functions.html#textattack.goal_functions.GoalFunction.get_output
[292]: api/goal_functions.html#textattack.goal_functions.GoalFunction.get_result
[293]: api/goal_functions.html#textattack.goal_functions.GoalFunction.get_results
[294]: api/goal_functions.html#textattack.goal_functions.GoalFunction.init_attack_example
[295]: api/goal_functions.html#logitsum
[296]: api/goal_functions.html#textattack.goal_functions.LogitSum
[297]: api/goal_functions.html#namedentityrecognition
[298]: api/goal_functions.html#textattack.goal_functions.NamedEntityRecognition
[299]: api/goal_functions.html#targetedstrict
[300]: api/goal_functions.html#textattack.goal_functions.TargetedStrict
[301]: api/goal_functions.html#targetedbonus
[302]: api/goal_functions.html#textattack.goal_functions.TargetedBonus
[303]: api/goal_functions.html#classificationgoalfunction
[304]: api/goal_functions.html#textattack.goal_functions.classification.ClassificationGoalFunction
[305]: api/goal_functions.html#textattack.goal_functions.classification.ClassificationGoalFunction.e
xtra_repr_keys
[306]: api/goal_functions.html#targetedclassification
[307]: api/goal_functions.html#textattack.goal_functions.classification.TargetedClassification
[308]: api/goal_functions.html#textattack.goal_functions.classification.TargetedClassification.extra
_repr_keys
[309]: api/goal_functions.html#untargetedclassification
[310]: api/goal_functions.html#textattack.goal_functions.classification.UntargetedClassification
[311]: api/goal_functions.html#inputreduction
[312]: api/goal_functions.html#textattack.goal_functions.classification.InputReduction
[313]: api/goal_functions.html#textattack.goal_functions.classification.InputReduction.extra_repr_ke
ys
[314]: api/goal_functions.html#texttotextgoalfunction
[315]: api/goal_functions.html#textattack.goal_functions.text.TextToTextGoalFunction
[316]: api/goal_functions.html#minimizebleu
[317]: api/goal_functions.html#textattack.goal_functions.text.MinimizeBleu
[318]: api/goal_functions.html#textattack.goal_functions.text.MinimizeBleu.extra_repr_keys
[319]: api/goal_functions.html#nonoverlappingoutput
[320]: api/goal_functions.html#textattack.goal_functions.text.NonOverlappingOutput
[321]: api/goal_functions.html#maximizelevenshtein
[322]: api/goal_functions.html#textattack.goal_functions.text.MaximizeLevenshtein
[323]: api/goal_functions.html#textattack.goal_functions.text.MaximizeLevenshtein.extra_repr_keys
[324]: api/constraints.html
[325]: api/constraints.html#constraint
[326]: api/constraints.html#textattack.constraints.Constraint
[327]: api/constraints.html#textattack.constraints.Constraint.call_many
[328]: api/constraints.html#textattack.constraints.Constraint.check_compatibility
[329]: api/constraints.html#textattack.constraints.Constraint.extra_repr_keys
[330]: api/constraints.html#pretransformationconstraint
[331]: api/constraints.html#textattack.constraints.PreTransformationConstraint
[332]: api/constraints.html#textattack.constraints.PreTransformationConstraint.check_compatibility
[333]: api/constraints.html#textattack.constraints.PreTransformationConstraint.extra_repr_keys
[334]: api/transformations.html
[335]: api/transformations.html#transformation
[336]: api/transformations.html#textattack.transformations.Transformation
[337]: api/transformations.html#composite-transformation
[338]: api/transformations.html#textattack.transformations.CompositeTransformation
[339]: api/search_methods.html
[340]: api/search_methods.html#searchmethod
[341]: api/search_methods.html#textattack.search_methods.SearchMethod
[342]: api/search_methods.html#textattack.search_methods.SearchMethod.check_transformation_compatibi
lity
[343]: api/search_methods.html#textattack.search_methods.SearchMethod.perform_search
[344]: api/search_methods.html#textattack.search_methods.SearchMethod.is_black_box
[345]: api/search_methods.html#beamsearch
[346]: api/search_methods.html#textattack.search_methods.BeamSearch
[347]: api/search_methods.html#textattack.search_methods.BeamSearch.extra_repr_keys
[348]: api/search_methods.html#textattack.search_methods.BeamSearch.perform_search
[349]: api/search_methods.html#textattack.search_methods.BeamSearch.is_black_box
[350]: api/search_methods.html#greedysearch
[351]: api/search_methods.html#textattack.search_methods.GreedySearch
[352]: api/search_methods.html#textattack.search_methods.GreedySearch.extra_repr_keys
[353]: api/search_methods.html#greedywordswapwir
[354]: api/search_methods.html#textattack.search_methods.GreedyWordSwapWIR
[355]: api/search_methods.html#textattack.search_methods.GreedyWordSwapWIR.check_transformation_comp
atibility
[356]: api/search_methods.html#textattack.search_methods.GreedyWordSwapWIR.extra_repr_keys
[357]: api/search_methods.html#textattack.search_methods.GreedyWordSwapWIR.perform_search
[358]: api/search_methods.html#textattack.search_methods.GreedyWordSwapWIR.is_black_box
[359]: api/search_methods.html#alzantotgeneticalgorithm
[360]: api/search_methods.html#textattack.search_methods.AlzantotGeneticAlgorithm
[361]: api/search_methods.html#improvedgeneticalgorithm
[362]: api/search_methods.html#textattack.search_methods.ImprovedGeneticAlgorithm
[363]: api/search_methods.html#textattack.search_methods.ImprovedGeneticAlgorithm.extra_repr_keys
[364]: api/search_methods.html#particleswarmoptimization
[365]: api/search_methods.html#textattack.search_methods.ParticleSwarmOptimization
[366]: api/search_methods.html#textattack.search_methods.ParticleSwarmOptimization.check_transformat
ion_compatibility
[367]: api/search_methods.html#textattack.search_methods.ParticleSwarmOptimization.extra_repr_keys
[368]: api/search_methods.html#textattack.search_methods.ParticleSwarmOptimization.perform_search
[369]: api/search_methods.html#textattack.search_methods.ParticleSwarmOptimization.is_black_box
[370]: api/search_methods.html#differentialevolution
[371]: api/search_methods.html#textattack.search_methods.DifferentialEvolution
[372]: api/search_methods.html#textattack.search_methods.DifferentialEvolution.check_transformation_
compatibility
[373]: api/search_methods.html#textattack.search_methods.DifferentialEvolution.extra_repr_keys
[374]: api/search_methods.html#textattack.search_methods.DifferentialEvolution.perform_search
[375]: api/search_methods.html#textattack.search_methods.DifferentialEvolution.is_black_box
[376]: apidoc/textattack.html
[377]: apidoc/textattack.html#subpackages
[378]: apidoc/textattack.attack_recipes.html
[379]: apidoc/textattack.attack_recipes.html#attack-recipes-package
[380]: apidoc/textattack.attack_recipes.html#module-textattack.attack_recipes.a2t_yoo_2021
[381]: apidoc/textattack.attack_recipes.html#a2t-a2t-attack-for-adversarial-training-recipe
[382]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021
[383]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021.build
[384]: apidoc/textattack.attack_recipes.html#attack-recipe-class
[385]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.attack_recipe.AttackRecipe
[386]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.attack_recipe.AttackRecipe.bu
ild
[387]: apidoc/textattack.attack_recipes.html#imperceptible-perturbations-algorithm
[388]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.bad_characters_2021.BadCharac
ters2021
[389]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.bad_characters_2021.BadCharac
ters2021.build
[390]: apidoc/textattack.attack_recipes.html#bae-bae-bert-based-adversarial-examples
[391]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.bae_garg_2019.BAEGarg2019
[392]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.bae_garg_2019.BAEGarg2019.bui
ld
[393]: apidoc/textattack.attack_recipes.html#bert-attack
[394]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.bert_attack_li_2020.BERTAttac
kLi2020
[395]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.bert_attack_li_2020.BERTAttac
kLi2020.build
[396]: apidoc/textattack.attack_recipes.html#checklist
[397]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.checklist_ribeiro_2020.CheckL
ist2020
[398]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.checklist_ribeiro_2020.CheckL
ist2020.build
[399]: apidoc/textattack.attack_recipes.html#attack-chinese-recipe
[400]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.chinese_recipe.ChineseRecipe
[401]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.chinese_recipe.ChineseRecipe.
build
[402]: apidoc/textattack.attack_recipes.html#clare-recipe
[403]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.clare_li_2020.CLARE2020
[404]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.clare_li_2020.CLARE2020.build
[405]: apidoc/textattack.attack_recipes.html#deepwordbug
[406]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.deepwordbug_gao_2018.DeepWord
BugGao2018
[407]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.deepwordbug_gao_2018.DeepWord
BugGao2018.build
[408]: apidoc/textattack.attack_recipes.html#faster-alzantot-genetic-algorithm
[409]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.faster_genetic_algorithm_jia_
2019.FasterGeneticAlgorithmJia2019
[410]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.faster_genetic_algorithm_jia_
2019.FasterGeneticAlgorithmJia2019.build
[411]: apidoc/textattack.attack_recipes.html#attack-french-recipe
[412]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.french_recipe.FrenchRecipe
[413]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.french_recipe.FrenchRecipe.bu
ild
[414]: apidoc/textattack.attack_recipes.html#alzantot-genetic-algorithm
[415]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.genetic_algorithm_alzantot_20
18.GeneticAlgorithmAlzantot2018
[416]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.genetic_algorithm_alzantot_20
18.GeneticAlgorithmAlzantot2018.build
[417]: apidoc/textattack.attack_recipes.html#hotflip
[418]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.hotflip_ebrahimi_2017.HotFlip
Ebrahimi2017
[419]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.hotflip_ebrahimi_2017.HotFlip
Ebrahimi2017.build
[420]: apidoc/textattack.attack_recipes.html#improved-genetic-algorithm
[421]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.iga_wang_2019.IGAWang2019
[422]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.iga_wang_2019.IGAWang2019.bui
ld
[423]: apidoc/textattack.attack_recipes.html#input-reduction
[424]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.input_reduction_feng_2018.Inp
utReductionFeng2018
[425]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.input_reduction_feng_2018.Inp
utReductionFeng2018.build
[426]: apidoc/textattack.attack_recipes.html#kuleshov2017
[427]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.kuleshov_2017.Kuleshov2017
[428]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.kuleshov_2017.Kuleshov2017.bu
ild
[429]: apidoc/textattack.attack_recipes.html#morpheus2020
[430]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.morpheus_tan_2020.MorpheusTan
2020
[431]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.morpheus_tan_2020.MorpheusTan
2020.build
[432]: apidoc/textattack.attack_recipes.html#pruthi2019-combating-with-robust-word-recognition
[433]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.pruthi_2019.Pruthi2019
[434]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.pruthi_2019.Pruthi2019.build
[435]: apidoc/textattack.attack_recipes.html#particle-swarm-optimization
[436]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.pso_zang_2020.PSOZang2020
[437]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.pso_zang_2020.PSOZang2020.bui
ld
[438]: apidoc/textattack.attack_recipes.html#pwws
[439]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.pwws_ren_2019.PWWSRen2019
[440]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.pwws_ren_2019.PWWSRen2019.bui
ld
[441]: apidoc/textattack.attack_recipes.html#seq2sick
[442]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.seq2sick_cheng_2018_blackbox.
Seq2SickCheng2018BlackBox
[443]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.seq2sick_cheng_2018_blackbox.
Seq2SickCheng2018BlackBox.build
[444]: apidoc/textattack.attack_recipes.html#attack-spanish-recipe
[445]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.spanish_recipe.SpanishRecipe
[446]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.spanish_recipe.SpanishRecipe.
build
[447]: apidoc/textattack.attack_recipes.html#textbugger
[448]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.textbugger_li_2018.TextBugger
Li2018
[449]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.textbugger_li_2018.TextBugger
Li2018.build
[450]: apidoc/textattack.attack_recipes.html#textfooler-is-bert-really-robust
[451]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.textfooler_jin_2019.TextFoole
rJin2019
[452]: apidoc/textattack.attack_recipes.html#textattack.attack_recipes.textfooler_jin_2019.TextFoole
rJin2019.build
[453]: apidoc/textattack.attack_results.html
[454]: apidoc/textattack.attack_results.html#id1
[455]: apidoc/textattack.attack_results.html#module-textattack.attack_results.attack_result
[456]: apidoc/textattack.attack_results.html#attackresult-class
[457]: apidoc/textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult
[458]: apidoc/textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.di
ff_color
[459]: apidoc/textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.go
al_function_result_str
[460]: apidoc/textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.or
iginal_text
[461]: apidoc/textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.pe
rturbed_text
[462]: apidoc/textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.st
r_lines
[463]: apidoc/textattack.attack_results.html#failedattackresult-class
[464]: apidoc/textattack.attack_results.html#textattack.attack_results.failed_attack_result.FailedAt
tackResult
[465]: apidoc/textattack.attack_results.html#textattack.attack_results.failed_attack_result.FailedAt
tackResult.goal_function_result_str
[466]: apidoc/textattack.attack_results.html#textattack.attack_results.failed_attack_result.FailedAt
tackResult.str_lines
[467]: apidoc/textattack.attack_results.html#maximizedattackresult-class
[468]: apidoc/textattack.attack_results.html#textattack.attack_results.maximized_attack_result.Maxim
izedAttackResult
[469]: apidoc/textattack.attack_results.html#skippedattackresult-class
[470]: apidoc/textattack.attack_results.html#textattack.attack_results.skipped_attack_result.Skipped
AttackResult
[471]: apidoc/textattack.attack_results.html#textattack.attack_results.skipped_attack_result.Skipped
AttackResult.goal_function_result_str
[472]: apidoc/textattack.attack_results.html#textattack.attack_results.skipped_attack_result.Skipped
AttackResult.str_lines
[473]: apidoc/textattack.attack_results.html#successfulattackresult-class
[474]: apidoc/textattack.attack_results.html#textattack.attack_results.successful_attack_result.Succ
essfulAttackResult
[475]: apidoc/textattack.augmentation.html
[476]: apidoc/textattack.augmentation.html#augmentation
[477]: apidoc/textattack.augmentation.html#module-textattack.augmentation.augmenter
[478]: apidoc/textattack.augmentation.html#augmenter-class
[479]: apidoc/textattack.augmentation.html#textattack.augmentation.augmenter.AugmentationResult
[480]: apidoc/textattack.augmentation.html#textattack.augmentation.augmenter.AugmentationResult.temp
Result
[481]: apidoc/textattack.augmentation.html#textattack.augmentation.augmenter.Augmenter
[482]: apidoc/textattack.augmentation.html#textattack.augmentation.augmenter.Augmenter.augment
[483]: apidoc/textattack.augmentation.html#textattack.augmentation.augmenter.Augmenter.augment_many
[484]: apidoc/textattack.augmentation.html#textattack.augmentation.augmenter.Augmenter.augment_text_
with_ids
[485]: apidoc/textattack.augmentation.html#augmenter-recipes
[486]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.BackTranscriptionAugmente
r
[487]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.BackTranslationAugmenter
[488]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.CLAREAugmenter
[489]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.CharSwapAugmenter
[490]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.CheckListAugmenter
[491]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.DeletionAugmenter
[492]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.EasyDataAugmenter
[493]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.EasyDataAugmenter.augment
[494]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.EmbeddingAugmenter
[495]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.SwapAugmenter
[496]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.SynonymInsertionAugmenter
[497]: apidoc/textattack.augmentation.html#textattack.augmentation.recipes.WordNetAugmenter
[498]: apidoc/textattack.commands.html
[499]: apidoc/textattack.commands.html#id1
[500]: apidoc/textattack.commands.html#module-textattack.commands.attack_command
[501]: apidoc/textattack.commands.html#attackcommand-class
[502]: apidoc/textattack.commands.html#textattack.commands.attack_command.AttackCommand
[503]: apidoc/textattack.commands.html#textattack.commands.attack_command.AttackCommand.register_sub
command
[504]: apidoc/textattack.commands.html#textattack.commands.attack_command.AttackCommand.run
[505]: apidoc/textattack.commands.html#attackresumecommand-class
[506]: apidoc/textattack.commands.html#textattack.commands.attack_resume_command.AttackResumeCommand
[507]: apidoc/textattack.commands.html#textattack.commands.attack_resume_command.AttackResumeCommand
.register_subcommand
[508]: apidoc/textattack.commands.html#textattack.commands.attack_resume_command.AttackResumeCommand
.run
[509]: apidoc/textattack.commands.html#augmentcommand-class
[510]: apidoc/textattack.commands.html#textattack.commands.augment_command.AugmentCommand
[511]: apidoc/textattack.commands.html#textattack.commands.augment_command.AugmentCommand.register_s
ubcommand
[512]: apidoc/textattack.commands.html#textattack.commands.augment_command.AugmentCommand.run
[513]: apidoc/textattack.commands.html#benchmarkrecipecommand-class
[514]: apidoc/textattack.commands.html#textattack.commands.benchmark_recipe_command.BenchmarkRecipeC
ommand
[515]: apidoc/textattack.commands.html#textattack.commands.benchmark_recipe_command.BenchmarkRecipeC
ommand.register_subcommand
[516]: apidoc/textattack.commands.html#textattack.commands.benchmark_recipe_command.BenchmarkRecipeC
ommand.run
[517]: apidoc/textattack.commands.html#evalmodelcommand-class
[518]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand
[519]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand.get_p
reds
[520]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand.regis
ter_subcommand
[521]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand.run
[522]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand.test_
model_on_dataset
[523]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs
[524]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs.batch_si
ze
[525]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs.num_exam
ples
[526]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs.num_exam
ples_offset
[527]: apidoc/textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs.random_s
eed
[528]: apidoc/textattack.commands.html#listthingscommand-class
[529]: apidoc/textattack.commands.html#textattack.commands.list_things_command.ListThingsCommand
[530]: apidoc/textattack.commands.html#textattack.commands.list_things_command.ListThingsCommand.reg
ister_subcommand
[531]: apidoc/textattack.commands.html#textattack.commands.list_things_command.ListThingsCommand.run
[532]: apidoc/textattack.commands.html#textattack.commands.list_things_command.ListThingsCommand.thi
ngs
[533]: apidoc/textattack.commands.html#peekdatasetcommand-class
[534]: apidoc/textattack.commands.html#textattack.commands.peek_dataset_command.PeekDatasetCommand
[535]: apidoc/textattack.commands.html#textattack.commands.peek_dataset_command.PeekDatasetCommand.r
egister_subcommand
[536]: apidoc/textattack.commands.html#textattack.commands.peek_dataset_command.PeekDatasetCommand.r
un
[537]: apidoc/textattack.commands.html#textattack-cli-main-class
[538]: apidoc/textattack.commands.html#textattack.commands.textattack_cli.main
[539]: apidoc/textattack.commands.html#textattack.commands.textattack_command.TextAttackCommand
[540]: apidoc/textattack.commands.html#textattack.commands.textattack_command.TextAttackCommand.regi
ster_subcommand
[541]: apidoc/textattack.commands.html#textattack.commands.textattack_command.TextAttackCommand.run
[542]: apidoc/textattack.commands.html#trainmodelcommand-class
[543]: apidoc/textattack.commands.html#textattack.commands.train_model_command.TrainModelCommand
[544]: apidoc/textattack.commands.html#textattack.commands.train_model_command.TrainModelCommand.reg
ister_subcommand
[545]: apidoc/textattack.commands.html#textattack.commands.train_model_command.TrainModelCommand.run
[546]: apidoc/textattack.constraints.html
[547]: apidoc/textattack.constraints.html#constraints
[548]: apidoc/textattack.constraints.html#subpackages
[549]: apidoc/textattack.constraints.grammaticality.html
[550]: apidoc/textattack.constraints.grammaticality.html#grammaticality
[551]: apidoc/textattack.constraints.grammaticality.html#subpackages
[552]: apidoc/textattack.constraints.grammaticality.language_models.html
[553]: apidoc/textattack.constraints.grammaticality.html#module-textattack.constraints.grammaticalit
y.cola
[554]: apidoc/textattack.constraints.grammaticality.html#cola-for-grammaticality
[555]: apidoc/textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.cola.
COLA
[556]: apidoc/textattack.constraints.grammaticality.html#languagetool-grammar-checker
[557]: apidoc/textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.langu
age_tool.LanguageTool
[558]: apidoc/textattack.constraints.grammaticality.html#part-of-speech-constraint
[559]: apidoc/textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.part_
of_speech.PartOfSpeech
[560]: apidoc/textattack.constraints.overlap.html
[561]: apidoc/textattack.constraints.overlap.html#overlap-constraints
[562]: apidoc/textattack.constraints.overlap.html#module-textattack.constraints.overlap.bleu_score
[563]: apidoc/textattack.constraints.overlap.html#bleu-constraints
[564]: apidoc/textattack.constraints.overlap.html#textattack.constraints.overlap.bleu_score.BLEU
[565]: apidoc/textattack.constraints.overlap.html#chrf-constraints
[566]: apidoc/textattack.constraints.overlap.html#textattack.constraints.overlap.chrf_score.chrF
[567]: apidoc/textattack.constraints.overlap.html#edit-distance-constraints
[568]: apidoc/textattack.constraints.overlap.html#textattack.constraints.overlap.levenshtein_edit_di
stance.LevenshteinEditDistance
[569]: apidoc/textattack.constraints.overlap.html#max-perturb-words-constraints
[570]: apidoc/textattack.constraints.overlap.html#textattack.constraints.overlap.max_words_perturbed
.MaxWordsPerturbed
[571]: apidoc/textattack.constraints.overlap.html#meteor-constraints
[572]: apidoc/textattack.constraints.overlap.html#textattack.constraints.overlap.meteor_score.METEOR
[573]: apidoc/textattack.constraints.pre_transformation.html
[574]: apidoc/textattack.constraints.pre_transformation.html#pre-transformation
[575]: apidoc/textattack.constraints.pre_transformation.html#module-textattack.constraints.pre_trans
formation.input_column_modification
[576]: apidoc/textattack.constraints.pre_transformation.html#input-column-modification
[577]: apidoc/textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformati
on.input_column_modification.InputColumnModification
[578]: apidoc/textattack.constraints.pre_transformation.html#max-modification-rate
[579]: apidoc/textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformati
on.max_modification_rate.MaxModificationRate
[580]: apidoc/textattack.constraints.pre_transformation.html#id2
[581]: apidoc/textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformati
on.max_num_words_modified.MaxNumWordsModified
[582]: apidoc/textattack.constraints.pre_transformation.html#max-word-index-modification
[583]: apidoc/textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformati
on.max_word_index_modification.MaxWordIndexModification
[584]: apidoc/textattack.constraints.pre_transformation.html#min-word-lenth
[585]: apidoc/textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformati
on.min_word_length.MinWordLength
[586]: apidoc/textattack.constraints.pre_transformation.html#repeat-modification
[587]: apidoc/textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformati
on.repeat_modification.RepeatModification
[588]: apidoc/textattack.constraints.pre_transformation.html#stopword-modification
[589]: apidoc/textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformati
on.stopword_modification.StopwordModification
[590]: apidoc/textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformati
on.unmodifiable_indices.UnmodifiableIndices
[591]: apidoc/textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformati
on.unmodifiable_phrases.UnmodifablePhrases
[592]: apidoc/textattack.constraints.semantics.html
[593]: apidoc/textattack.constraints.semantics.html#semantic-constraints
[594]: apidoc/textattack.constraints.semantics.html#subpackages
[595]: apidoc/textattack.constraints.semantics.sentence_encoders.html
[596]: apidoc/textattack.constraints.semantics.html#module-textattack.constraints.semantics.bert_sco
re
[597]: apidoc/textattack.constraints.semantics.html#bert-score
[598]: apidoc/textattack.constraints.semantics.html#textattack.constraints.semantics.bert_score.BERT
Score
[599]: apidoc/textattack.constraints.semantics.html#word-embedding-distance
[600]: apidoc/textattack.constraints.semantics.html#textattack.constraints.semantics.word_embedding_
distance.WordEmbeddingDistance
[601]: apidoc/textattack.constraints.html#module-textattack.constraints.constraint
[602]: apidoc/textattack.constraints.html#textattack-constraint-class
[603]: apidoc/textattack.constraints.html#textattack.constraints.constraint.Constraint
[604]: apidoc/textattack.constraints.html#textattack.constraints.constraint.Constraint.call_many
[605]: apidoc/textattack.constraints.html#textattack.constraints.constraint.Constraint.check_compati
bility
[606]: apidoc/textattack.constraints.html#textattack.constraints.constraint.Constraint.extra_repr_ke
ys
[607]: apidoc/textattack.constraints.html#pre-transformation-constraint-class
[608]: apidoc/textattack.constraints.html#textattack.constraints.pre_transformation_constraint.PreTr
ansformationConstraint
[609]: apidoc/textattack.constraints.html#textattack.constraints.pre_transformation_constraint.PreTr
ansformationConstraint.check_compatibility
[610]: apidoc/textattack.constraints.html#textattack.constraints.pre_transformation_constraint.PreTr
ansformationConstraint.extra_repr_keys
[611]: apidoc/textattack.datasets.html
[612]: apidoc/textattack.datasets.html#datasets-package
[613]: apidoc/textattack.datasets.html#subpackages
[614]: apidoc/textattack.datasets.helpers.html
[615]: apidoc/textattack.datasets.helpers.html#dataset-helpers
[616]: apidoc/textattack.datasets.helpers.html#module-textattack.datasets.helpers.ted_multi
[617]: apidoc/textattack.datasets.helpers.html#ted-multi-translationdataset-class
[618]: apidoc/textattack.datasets.helpers.html#textattack.datasets.helpers.ted_multi.TedMultiTransla
tionDataset
[619]: apidoc/textattack.datasets.html#module-textattack.datasets.dataset
[620]: apidoc/textattack.datasets.html#dataset-class
[621]: apidoc/textattack.datasets.html#textattack.datasets.dataset.Dataset
[622]: apidoc/textattack.datasets.html#textattack.datasets.dataset.Dataset.filter_by_labels_
[623]: apidoc/textattack.datasets.html#textattack.datasets.dataset.Dataset.shuffle
[624]: apidoc/textattack.datasets.html#huggingfacedataset-class
[625]: apidoc/textattack.datasets.html#textattack.datasets.huggingface_dataset.HuggingFaceDataset
[626]: apidoc/textattack.datasets.html#textattack.datasets.huggingface_dataset.HuggingFaceDataset.fi
lter_by_labels_
[627]: apidoc/textattack.datasets.html#textattack.datasets.huggingface_dataset.HuggingFaceDataset.sh
uffle
[628]: apidoc/textattack.datasets.html#textattack.datasets.huggingface_dataset.get_datasets_dataset_
columns
[629]: apidoc/textattack.goal_function_results.html
[630]: apidoc/textattack.goal_function_results.html#goal-function-result-package
[631]: apidoc/textattack.goal_function_results.html#subpackages
[632]: apidoc/textattack.goal_function_results.custom.html
[633]: apidoc/textattack.goal_function_results.custom.html#custom-goal-function-result-package
[634]: apidoc/textattack.goal_function_results.custom.html#module-textattack.goal_function_results.c
ustom.logit_sum_goal_function_result
[635]: apidoc/textattack.goal_function_results.custom.html#logitsumgoalfunctionresult-class
[636]: apidoc/textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.l
ogit_sum_goal_function_result.LogitSumGoalFunctionResult
[637]: apidoc/textattack.goal_function_results.custom.html#namedentityrecognitionoalfunctionresult-c
lass
[638]: apidoc/textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.n
amed_entity_recognition_goal_function_result.NamedEntityRecognitionGoalFunctionResult
[639]: apidoc/textattack.goal_function_results.custom.html#targetedbonusgoalfunctionresult-class
[640]: apidoc/textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.t
argeted_bonus_goal_function_result.TargetedBonusGoalFunctionResult
[641]: apidoc/textattack.goal_function_results.custom.html#targetedstrictgoalfunctionresult-class
[642]: apidoc/textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.t
argeted_strict_goal_function_result.TargetedStrictGoalFunctionResult
[643]: apidoc/textattack.goal_function_results.html#module-textattack.goal_function_results.classifi
cation_goal_function_result
[644]: apidoc/textattack.goal_function_results.html#classificationgoalfunctionresult-class
[645]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.classification_
goal_function_result.ClassificationGoalFunctionResult
[646]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.classification_
goal_function_result.ClassificationGoalFunctionResult.get_colored_output
[647]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.classification_
goal_function_result.ClassificationGoalFunctionResult.get_text_color_input
[648]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.classification_
goal_function_result.ClassificationGoalFunctionResult.get_text_color_perturbed
[649]: apidoc/textattack.goal_function_results.html#goalfunctionresult-class
[650]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.goal_function_r
esult.GoalFunctionResult
[651]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.goal_function_r
esult.GoalFunctionResult.get_colored_output
[652]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.goal_function_r
esult.GoalFunctionResult.get_text_color_input
[653]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.goal_function_r
esult.GoalFunctionResult.get_text_color_perturbed
[654]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.goal_function_r
esult.GoalFunctionResultStatus
[655]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.goal_function_r
esult.GoalFunctionResultStatus.MAXIMIZING
[656]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.goal_function_r
esult.GoalFunctionResultStatus.SEARCHING
[657]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.goal_function_r
esult.GoalFunctionResultStatus.SKIPPED
[658]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.goal_function_r
esult.GoalFunctionResultStatus.SUCCEEDED
[659]: apidoc/textattack.goal_function_results.html#texttotextgoalfunctionresult-class
[660]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.text_to_text_go
al_function_result.TextToTextGoalFunctionResult
[661]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.text_to_text_go
al_function_result.TextToTextGoalFunctionResult.get_colored_output
[662]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.text_to_text_go
al_function_result.TextToTextGoalFunctionResult.get_text_color_input
[663]: apidoc/textattack.goal_function_results.html#textattack.goal_function_results.text_to_text_go
al_function_result.TextToTextGoalFunctionResult.get_text_color_perturbed
[664]: apidoc/textattack.goal_functions.html
[665]: apidoc/textattack.goal_functions.html#goal-functions
[666]: apidoc/textattack.goal_functions.html#subpackages
[667]: apidoc/textattack.goal_functions.classification.html
[668]: apidoc/textattack.goal_functions.classification.html#goal-fucntion-for-classification
[669]: apidoc/textattack.goal_functions.classification.html#module-textattack.goal_functions.classif
ication.classification_goal_function
[670]: apidoc/textattack.goal_functions.classification.html#determine-for-if-an-attack-has-been-succ
essful-in-classification
[671]: apidoc/textattack.goal_functions.classification.html#textattack.goal_functions.classification
.classification_goal_function.ClassificationGoalFunction
[672]: apidoc/textattack.goal_functions.classification.html#determine-if-an-attack-has-been-successf
ul-in-hard-label-classficiation
[673]: apidoc/textattack.goal_functions.classification.html#textattack.goal_functions.classification
.hardlabel_classification.HardLabelClassification
[674]: apidoc/textattack.goal_functions.classification.html#determine-if-maintaining-the-same-predic
ted-label-input-reduction
[675]: apidoc/textattack.goal_functions.classification.html#textattack.goal_functions.classification
.input_reduction.InputReduction
[676]: apidoc/textattack.goal_functions.classification.html#determine-if-an-attack-has-been-successf
ul-in-targeted-classification
[677]: apidoc/textattack.goal_functions.classification.html#textattack.goal_functions.classification
.targeted_classification.TargetedClassification
[678]: apidoc/textattack.goal_functions.classification.html#determine-successful-in-untargeted-class
ification
[679]: apidoc/textattack.goal_functions.classification.html#textattack.goal_functions.classification
.untargeted_classification.UntargetedClassification
[680]: apidoc/textattack.goal_functions.custom.html
[681]: apidoc/textattack.goal_functions.custom.html#custom-goal-functions
[682]: apidoc/textattack.goal_functions.custom.html#module-textattack.goal_functions.custom.logit_su
m
[683]: apidoc/textattack.goal_functions.custom.html#goal-function-for-logit-sum
[684]: apidoc/textattack.goal_functions.custom.html#textattack.goal_functions.custom.logit_sum.Logit
Sum
[685]: apidoc/textattack.goal_functions.custom.html#goal-function-for-namedentityrecognition
[686]: apidoc/textattack.goal_functions.custom.html#textattack.goal_functions.custom.named_entity_re
cognition.NamedEntityRecognition
[687]: apidoc/textattack.goal_functions.custom.html#goal-function-for-targeted-classification-with-b
onus-score
[688]: apidoc/textattack.goal_functions.custom.html#textattack.goal_functions.custom.targeted_bonus.
TargetedBonus
[689]: apidoc/textattack.goal_functions.custom.html#goal-function-for-strict-targeted-classification
[690]: apidoc/textattack.goal_functions.custom.html#textattack.goal_functions.custom.targeted_strict
.TargetedStrict
[691]: apidoc/textattack.goal_functions.text.html
[692]: apidoc/textattack.goal_functions.text.html#goal-function-for-text-to-text-case
[693]: apidoc/textattack.goal_functions.text.html#module-textattack.goal_functions.text.maximize_lev
enshtein
[694]: apidoc/textattack.goal_functions.text.html#textattack.goal_functions.text.maximize_levenshtei
n.MaximizeLevenshtein
[695]: apidoc/textattack.goal_functions.text.html#goal-function-for-attempts-to-minimize-the-bleu-sc
ore
[696]: apidoc/textattack.goal_functions.text.html#textattack.goal_functions.text.minimize_bleu.Minim
izeBleu
[697]: apidoc/textattack.goal_functions.text.html#textattack.goal_functions.text.minimize_bleu.get_b
leu
[698]: apidoc/textattack.goal_functions.text.html#goal-function-for-seq2sick
[699]: apidoc/textattack.goal_functions.text.html#textattack.goal_functions.text.non_overlapping_out
put.NonOverlappingOutput
[700]: apidoc/textattack.goal_functions.text.html#textattack.goal_functions.text.non_overlapping_out
put.get_words_cached
[701]: apidoc/textattack.goal_functions.text.html#textattack.goal_functions.text.non_overlapping_out
put.word_difference_score
[702]: apidoc/textattack.goal_functions.text.html#goal-function-for-texttotext
[703]: apidoc/textattack.goal_functions.text.html#textattack.goal_functions.text.text_to_text_goal_f
unction.TextToTextGoalFunction
[704]: apidoc/textattack.goal_functions.html#module-textattack.goal_functions.goal_function
[705]: apidoc/textattack.goal_functions.html#goalfunction-class
[706]: apidoc/textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction
[707]: apidoc/textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.cl
ear_cache
[708]: apidoc/textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.ex
tra_repr_keys
[709]: apidoc/textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.ge
t_output
[710]: apidoc/textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.ge
t_result
[711]: apidoc/textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.ge
t_results
[712]: apidoc/textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.in
it_attack_example
[713]: apidoc/textattack.llms.html
[714]: apidoc/textattack.llms.html#large-language-models
[715]: apidoc/textattack.llms.html#module-textattack.llms.chat_gpt_wrapper
[716]: apidoc/textattack.llms.html#textattack.llms.chat_gpt_wrapper.ChatGptWrapper
[717]: apidoc/textattack.llms.html#textattack.llms.huggingface_llm_wrapper.HuggingFaceLLMWrapper
[718]: apidoc/textattack.loggers.html
[719]: apidoc/textattack.loggers.html#misc-loggers-loggers-track-visualize-and-export-attack-results
[720]: apidoc/textattack.loggers.html#module-textattack.loggers.attack_log_manager
[721]: apidoc/textattack.loggers.html#managing-attack-logs
[722]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager
[723]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.add_out
put_csv
[724]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.add_out
put_file
[725]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.add_out
put_summary_json
[726]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.disable
_color
[727]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.enable_
stdout
[728]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.enable_
visdom
[729]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.enable_
wandb
[730]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.flush
[731]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_att
ack_details
[732]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_res
ult
[733]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_res
ults
[734]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_sep
[735]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_sum
mary
[736]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_sum
mary_rows
[737]: apidoc/textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.metrics
[738]: apidoc/textattack.loggers.html#attack-logs-to-csv
[739]: apidoc/textattack.loggers.html#textattack.loggers.csv_logger.CSVLogger
[740]: apidoc/textattack.loggers.html#textattack.loggers.csv_logger.CSVLogger.close
[741]: apidoc/textattack.loggers.html#textattack.loggers.csv_logger.CSVLogger.flush
[742]: apidoc/textattack.loggers.html#textattack.loggers.csv_logger.CSVLogger.log_attack_result
[743]: apidoc/textattack.loggers.html#attack-logs-to-file
[744]: apidoc/textattack.loggers.html#textattack.loggers.file_logger.FileLogger
[745]: apidoc/textattack.loggers.html#textattack.loggers.file_logger.FileLogger.close
[746]: apidoc/textattack.loggers.html#textattack.loggers.file_logger.FileLogger.flush
[747]: apidoc/textattack.loggers.html#textattack.loggers.file_logger.FileLogger.log_attack_result
[748]: apidoc/textattack.loggers.html#textattack.loggers.file_logger.FileLogger.log_sep
[749]: apidoc/textattack.loggers.html#textattack.loggers.file_logger.FileLogger.log_summary_rows
[750]: apidoc/textattack.loggers.html#attack-summary-results-logs-to-json
[751]: apidoc/textattack.loggers.html#textattack.loggers.json_summary_logger.JsonSummaryLogger
[752]: apidoc/textattack.loggers.html#textattack.loggers.json_summary_logger.JsonSummaryLogger.close
[753]: apidoc/textattack.loggers.html#textattack.loggers.json_summary_logger.JsonSummaryLogger.flush
[754]: apidoc/textattack.loggers.html#textattack.loggers.json_summary_logger.JsonSummaryLogger.log_s
ummary_rows
[755]: apidoc/textattack.loggers.html#attack-logger-wrapper
[756]: apidoc/textattack.loggers.html#textattack.loggers.logger.Logger
[757]: apidoc/textattack.loggers.html#textattack.loggers.logger.Logger.close
[758]: apidoc/textattack.loggers.html#textattack.loggers.logger.Logger.flush
[759]: apidoc/textattack.loggers.html#textattack.loggers.logger.Logger.log_attack_result
[760]: apidoc/textattack.loggers.html#textattack.loggers.logger.Logger.log_hist
[761]: apidoc/textattack.loggers.html#textattack.loggers.logger.Logger.log_sep
[762]: apidoc/textattack.loggers.html#textattack.loggers.logger.Logger.log_summary_rows
[763]: apidoc/textattack.loggers.html#attack-logs-to-visdom
[764]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger
[765]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.bar
[766]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.flush
[767]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.hist
[768]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.log_attack_resul
t
[769]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.log_hist
[770]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.log_summary_rows
[771]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.table
[772]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.text
[773]: apidoc/textattack.loggers.html#textattack.loggers.visdom_logger.port_is_open
[774]: apidoc/textattack.loggers.html#attack-logs-to-wandb
[775]: apidoc/textattack.loggers.html#textattack.loggers.weights_and_biases_logger.WeightsAndBiasesL
ogger
[776]: apidoc/textattack.loggers.html#textattack.loggers.weights_and_biases_logger.WeightsAndBiasesL
ogger.log_attack_result
[777]: apidoc/textattack.loggers.html#textattack.loggers.weights_and_biases_logger.WeightsAndBiasesL
ogger.log_sep
[778]: apidoc/textattack.loggers.html#textattack.loggers.weights_and_biases_logger.WeightsAndBiasesL
ogger.log_summary_rows
[779]: apidoc/textattack.metrics.html
[780]: apidoc/textattack.metrics.html#metrics-package-to-calculate-advanced-metrics-for-evaluting-at
tacks-and-augmented-text
[781]: apidoc/textattack.metrics.html#subpackages
[782]: apidoc/textattack.metrics.attack_metrics.html
[783]: apidoc/textattack.metrics.attack_metrics.html#attack-metrics-package
[784]: apidoc/textattack.metrics.attack_metrics.html#module-textattack.metrics.attack_metrics.attack
_queries
[785]: apidoc/textattack.metrics.attack_metrics.html#metrics-on-attackqueries
[786]: apidoc/textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_querie
s.AttackQueries
[787]: apidoc/textattack.metrics.attack_metrics.html#metrics-on-attacksuccessrate
[788]: apidoc/textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_succes
s_rate.AttackSuccessRate
[789]: apidoc/textattack.metrics.attack_metrics.html#metrics-on-perturbed-words
[790]: apidoc/textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.words_perturb
ed.WordsPerturbed
[791]: apidoc/textattack.metrics.quality_metrics.html
[792]: apidoc/textattack.metrics.quality_metrics.html#metrics-on-quality-package
[793]: apidoc/textattack.metrics.quality_metrics.html#module-textattack.metrics.quality_metrics.bert
_score
[794]: apidoc/textattack.metrics.quality_metrics.html#bertscoremetric-class
[795]: apidoc/textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.bert_score.
BERTScoreMetric
[796]: apidoc/textattack.metrics.quality_metrics.html#meteormetric-class
[797]: apidoc/textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.meteor_scor
e.MeteorMetric
[798]: apidoc/textattack.metrics.quality_metrics.html#perplexity-metric
[799]: apidoc/textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.perplexity.
Perplexity
[800]: apidoc/textattack.metrics.quality_metrics.html#usemetric-class
[801]: apidoc/textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.sentence_be
rt.SBERTMetric
[802]: apidoc/textattack.metrics.quality_metrics.html#id1
[803]: apidoc/textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.use.USEMetr
ic
[804]: apidoc/textattack.metrics.html#module-textattack.metrics.metric
[805]: apidoc/textattack.metrics.html#metric-class
[806]: apidoc/textattack.metrics.html#textattack.metrics.metric.Metric
[807]: apidoc/textattack.metrics.html#textattack.metrics.metric.Metric.calculate
[808]: apidoc/textattack.metrics.html#attack-metric-quality-recipes
[809]: apidoc/textattack.metrics.html#textattack.metrics.recipe.AdvancedAttackMetric
[810]: apidoc/textattack.metrics.html#textattack.metrics.recipe.AdvancedAttackMetric.calculate
[811]: apidoc/textattack.models.html
[812]: apidoc/textattack.models.html#models
[813]: apidoc/textattack.models.html#models-user-specified
[814]: apidoc/textattack.models.html#models-pre-trained
[815]: apidoc/textattack.models.html#model-wrappers
[816]: apidoc/textattack.models.html#subpackages
[817]: apidoc/textattack.models.helpers.html
[818]: apidoc/textattack.models.helpers.html#moderl-helpers
[819]: apidoc/textattack.models.helpers.html#module-textattack.models.helpers.glove_embedding_layer
[820]: apidoc/textattack.models.helpers.html#glove-embedding
[821]: apidoc/textattack.models.helpers.html#textattack.models.helpers.glove_embedding_layer.Embeddi
ngLayer
[822]: apidoc/textattack.models.helpers.html#textattack.models.helpers.glove_embedding_layer.GloveEm
beddingLayer
[823]: apidoc/textattack.models.helpers.html#lstm-4-classification
[824]: apidoc/textattack.models.helpers.html#textattack.models.helpers.lstm_for_classification.LSTMF
orClassification
[825]: apidoc/textattack.models.helpers.html#t5-model-trained-to-generate-text-from-text
[826]: apidoc/textattack.models.helpers.html#textattack.models.helpers.t5_for_text_to_text.T5ForText
ToText
[827]: apidoc/textattack.models.helpers.html#util-function-for-model-wrapper
[828]: apidoc/textattack.models.helpers.html#textattack.models.helpers.utils.load_cached_state_dict
[829]: apidoc/textattack.models.helpers.html#word-cnn-for-classification
[830]: apidoc/textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.C
NNTextLayer
[831]: apidoc/textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.W
ordCNNForClassification
[832]: apidoc/textattack.models.tokenizers.html
[833]: apidoc/textattack.models.tokenizers.html#tokenizers-for-model-wrapper
[834]: apidoc/textattack.models.tokenizers.html#module-textattack.models.tokenizers.glove_tokenizer
[835]: apidoc/textattack.models.tokenizers.html#glove-tokenizer
[836]: apidoc/textattack.models.tokenizers.html#textattack.models.tokenizers.glove_tokenizer.GloveTo
kenizer
[837]: apidoc/textattack.models.tokenizers.html#textattack.models.tokenizers.glove_tokenizer.WordLev
elTokenizer
[838]: apidoc/textattack.models.tokenizers.html#t5-tokenizer
[839]: apidoc/textattack.models.tokenizers.html#textattack.models.tokenizers.t5_tokenizer.T5Tokenize
r
[840]: apidoc/textattack.models.wrappers.html
[841]: apidoc/textattack.models.wrappers.html#model-wrappers-package
[842]: apidoc/textattack.models.wrappers.html#module-textattack.models.wrappers.huggingface_model_wr
apper
[843]: apidoc/textattack.models.wrappers.html#huggingface-model-wrapper
[844]: apidoc/textattack.models.wrappers.html#textattack.models.wrappers.huggingface_model_wrapper.H
uggingFaceModelWrapper
[845]: apidoc/textattack.models.wrappers.html#modelwrapper-class
[846]: apidoc/textattack.models.wrappers.html#textattack.models.wrappers.model_wrapper.ModelWrapper
[847]: apidoc/textattack.models.wrappers.html#pytorch-model-wrapper
[848]: apidoc/textattack.models.wrappers.html#textattack.models.wrappers.pytorch_model_wrapper.PyTor
chModelWrapper
[849]: apidoc/textattack.models.wrappers.html#scikit-learn-model-wrapper
[850]: apidoc/textattack.models.wrappers.html#textattack.models.wrappers.sklearn_model_wrapper.Sklea
rnModelWrapper
[851]: apidoc/textattack.models.wrappers.html#tensorflow-model-wrapper
[852]: apidoc/textattack.models.wrappers.html#textattack.models.wrappers.tensorflow_model_wrapper.Te
nsorFlowModelWrapper
[853]: apidoc/textattack.prompt_augmentation.html
[854]: apidoc/textattack.prompt_augmentation.html#prompt-augmentation
[855]: apidoc/textattack.prompt_augmentation.html#module-textattack.prompt_augmentation.prompt_augme
ntation_pipeline
[856]: apidoc/textattack.prompt_augmentation.html#textattack.prompt_augmentation.prompt_augmentation
_pipeline.PromptAugmentationPipeline
[857]: apidoc/textattack.search_methods.html
[858]: apidoc/textattack.search_methods.html#search-methods
[859]: apidoc/textattack.search_methods.html#module-textattack.search_methods.alzantot_genetic_algor
ithm
[860]: apidoc/textattack.search_methods.html#reimplementation-of-search-method-from-generating-natur
al-language-adversarial-examples
[861]: apidoc/textattack.search_methods.html#textattack.search_methods.alzantot_genetic_algorithm.Al
zantotGeneticAlgorithm
[862]: apidoc/textattack.search_methods.html#beam-search
[863]: apidoc/textattack.search_methods.html#textattack.search_methods.beam_search.BeamSearch
[864]: apidoc/textattack.search_methods.html#textattack.search_methods.beam_search.BeamSearch.extra_
repr_keys
[865]: apidoc/textattack.search_methods.html#textattack.search_methods.beam_search.BeamSearch.perfor
m_search
[866]: apidoc/textattack.search_methods.html#textattack.search_methods.beam_search.BeamSearch.is_bla
ck_box
[867]: apidoc/textattack.search_methods.html#textattack.search_methods.differential_evolution.Differ
entialEvolution
[868]: apidoc/textattack.search_methods.html#textattack.search_methods.differential_evolution.Differ
entialEvolution.check_transformation_compatibility
[869]: apidoc/textattack.search_methods.html#textattack.search_methods.differential_evolution.Differ
entialEvolution.extra_repr_keys
[870]: apidoc/textattack.search_methods.html#textattack.search_methods.differential_evolution.Differ
entialEvolution.perform_search
[871]: apidoc/textattack.search_methods.html#textattack.search_methods.differential_evolution.Differ
entialEvolution.is_black_box
[872]: apidoc/textattack.search_methods.html#genetic-algorithm-word-swap
[873]: apidoc/textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgo
rithm
[874]: apidoc/textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgo
rithm.check_transformation_compatibility
[875]: apidoc/textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgo
rithm.extra_repr_keys
[876]: apidoc/textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgo
rithm.perform_search
[877]: apidoc/textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgo
rithm.is_black_box
[878]: apidoc/textattack.search_methods.html#greedy-search
[879]: apidoc/textattack.search_methods.html#textattack.search_methods.greedy_search.GreedySearch
[880]: apidoc/textattack.search_methods.html#textattack.search_methods.greedy_search.GreedySearch.ex
tra_repr_keys
[881]: apidoc/textattack.search_methods.html#greedy-word-swap-with-word-importance-ranking
[882]: apidoc/textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWo
rdSwapWIR
[883]: apidoc/textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWo
rdSwapWIR.check_transformation_compatibility
[884]: apidoc/textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWo
rdSwapWIR.extra_repr_keys
[885]: apidoc/textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWo
rdSwapWIR.perform_search
[886]: apidoc/textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWo
rdSwapWIR.is_black_box
[887]: apidoc/textattack.search_methods.html#reimplementation-of-search-method-from-xiaosen-wang-hao
-jin-kun-he-2019
[888]: apidoc/textattack.search_methods.html#textattack.search_methods.improved_genetic_algorithm.Im
provedGeneticAlgorithm
[889]: apidoc/textattack.search_methods.html#textattack.search_methods.improved_genetic_algorithm.Im
provedGeneticAlgorithm.extra_repr_keys
[890]: apidoc/textattack.search_methods.html#particle-swarm-optimization
[891]: apidoc/textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.P
articleSwarmOptimization
[892]: apidoc/textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.P
articleSwarmOptimization.check_transformation_compatibility
[893]: apidoc/textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.P
articleSwarmOptimization.extra_repr_keys
[894]: apidoc/textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.P
articleSwarmOptimization.perform_search
[895]: apidoc/textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.P
articleSwarmOptimization.is_black_box
[896]: apidoc/textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.n
ormalize
[897]: apidoc/textattack.search_methods.html#population-based-search-abstract-class
[898]: apidoc/textattack.search_methods.html#textattack.search_methods.population_based_search.Popul
ationBasedSearch
[899]: apidoc/textattack.search_methods.html#textattack.search_methods.population_based_search.Popul
ationMember
[900]: apidoc/textattack.search_methods.html#textattack.search_methods.population_based_search.Popul
ationMember.num_words
[901]: apidoc/textattack.search_methods.html#textattack.search_methods.population_based_search.Popul
ationMember.score
[902]: apidoc/textattack.search_methods.html#textattack.search_methods.population_based_search.Popul
ationMember.words
[903]: apidoc/textattack.search_methods.html#search-method-abstract-class
[904]: apidoc/textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod
[905]: apidoc/textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod.ch
eck_transformation_compatibility
[906]: apidoc/textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod.ge
t_victim_model
[907]: apidoc/textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod.pe
rform_search
[908]: apidoc/textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod.is
_black_box
[909]: apidoc/textattack.shared.html
[910]: apidoc/textattack.shared.html#shared-textattack-functions
[911]: apidoc/textattack.shared.html#subpackages
[912]: apidoc/textattack.shared.utils.html
[913]: apidoc/textattack.shared.utils.html#module-textattack.shared.utils.importing
[914]: apidoc/textattack.shared.utils.html#textattack.shared.utils.importing.LazyLoader
[915]: apidoc/textattack.shared.utils.html#textattack.shared.utils.importing.load_module_from_file
[916]: apidoc/textattack.shared.utils.html#textattack.shared.utils.install.download_from_s3
[917]: apidoc/textattack.shared.utils.html#textattack.shared.utils.install.download_from_url
[918]: apidoc/textattack.shared.utils.html#textattack.shared.utils.install.http_get
[919]: apidoc/textattack.shared.utils.html#textattack.shared.utils.install.path_in_cache
[920]: apidoc/textattack.shared.utils.html#textattack.shared.utils.install.s3_url
[921]: apidoc/textattack.shared.utils.html#textattack.shared.utils.install.set_cache_dir
[922]: apidoc/textattack.shared.utils.html#textattack.shared.utils.install.unzip_file
[923]: apidoc/textattack.shared.utils.html#textattack.shared.utils.misc.get_textattack_model_num_lab
els
[924]: apidoc/textattack.shared.utils.html#textattack.shared.utils.misc.hashable
[925]: apidoc/textattack.shared.utils.html#textattack.shared.utils.misc.html_style_from_dict
[926]: apidoc/textattack.shared.utils.html#textattack.shared.utils.misc.html_table_from_rows
[927]: apidoc/textattack.shared.utils.html#textattack.shared.utils.misc.load_textattack_model_from_p
ath
[928]: apidoc/textattack.shared.utils.html#textattack.shared.utils.misc.set_seed
[929]: apidoc/textattack.shared.utils.html#textattack.shared.utils.misc.sigmoid
[930]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES
[931]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.ReprMixin
[932]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.TextAttackFlairTokenizer
[933]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.add_indent
[934]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.check_if_punctuations
[935]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.check_if_subword
[936]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.color_from_label
[937]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.color_from_output
[938]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.color_text
[939]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.default_class_repr
[940]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.flair_tag
[941]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.has_letter
[942]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.is_one_word
[943]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.process_label_name
[944]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.strip_BPE_artifacts
[945]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.words_from_text
[946]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.zip_flair_result
[947]: apidoc/textattack.shared.utils.html#textattack.shared.utils.strings.zip_stanza_result
[948]: apidoc/textattack.shared.utils.html#textattack.shared.utils.tensor.batch_model_predict
[949]: apidoc/textattack.shared.html#module-textattack.shared.attacked_text
[950]: apidoc/textattack.shared.html#attacked-text-class
[951]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText
[952]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.align_with_model_t
okens
[953]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.all_words_diff
[954]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.convert_from_origi
nal_idxs
[955]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.delete_word_at_ind
ex
[956]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.first_word_diff
[957]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.first_word_diff_in
dex
[958]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.free_memory
[959]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.generate_new_attac
ked_text
[960]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.get_deletion_indic
es
[961]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.insert_text_after_
word_index
[962]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.insert_text_before
_word_index
[963]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.ith_word_diff
[964]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.ner_of_word_index
[965]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.pos_of_word_index
[966]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.printable_text
[967]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.replace_word_at_in
dex
[968]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.replace_words_at_i
ndices
[969]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.text_after_word_in
dex
[970]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.text_until_word_in
dex
[971]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.text_window_around
_index
[972]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.words_diff_num
[973]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.words_diff_ratio
[974]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.SPLIT_TOKEN
[975]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.column_labels
[976]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.newly_swapped_word
s
[977]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.num_words
[978]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.text
[979]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.tokenizer_input
[980]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.words
[981]: apidoc/textattack.shared.html#textattack.shared.attacked_text.AttackedText.words_per_input
[982]: apidoc/textattack.shared.html#misc-checkpoints
[983]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint
[984]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.load
[985]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.save
[986]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.dataset_offset
[987]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.datetime
[988]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_failed_attack
s
[989]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_maximized_att
acks
[990]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_remaining_att
acks
[991]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_skipped_attac
ks
[992]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_successful_at
tacks
[993]: apidoc/textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.results_count
[994]: apidoc/textattack.shared.html#shared-data-fields
[995]: apidoc/textattack.shared.html#misc-validators
[996]: apidoc/textattack.shared.html#textattack.shared.validators.transformation_consists_of
[997]: apidoc/textattack.shared.html#textattack.shared.validators.transformation_consists_of_word_sw
aps
[998]: apidoc/textattack.shared.html#textattack.shared.validators.transformation_consists_of_word_sw
aps_and_deletions
[999]: apidoc/textattack.shared.html#textattack.shared.validators.transformation_consists_of_word_sw
aps_differential_evolution
[1000]: apidoc/textattack.shared.html#textattack.shared.validators.validate_model_goal_function_comp
atibility
[1001]: apidoc/textattack.shared.html#textattack.shared.validators.validate_model_gradient_word_swap
_compatibility
[1002]: apidoc/textattack.shared.html#shared-loads-word-embeddings-and-related-distances
[1003]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding
[1004]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.get_co
s_sim
[1005]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.get_ms
e_dist
[1006]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.index2
word
[1007]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.neares
t_neighbours
[1008]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.word2i
ndex
[1009]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding
[1010]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.get_cos_
sim
[1011]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.get_mse_
dist
[1012]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.index2wo
rd
[1013]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.nearest_
neighbours
[1014]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.word2ind
ex
[1015]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding
[1016]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.counterfitted_
GLOVE_embedding
[1017]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.get_cos_sim
[1018]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.get_mse_dist
[1019]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.index2word
[1020]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.nearest_neighb
ours
[1021]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.word2index
[1022]: apidoc/textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.PATH
[1023]: apidoc/textattack.transformations.html
[1024]: apidoc/textattack.transformations.html#transformations
[1025]: apidoc/textattack.transformations.html#subpackages
[1026]: apidoc/textattack.transformations.sentence_transformations.html
[1027]: apidoc/textattack.transformations.sentence_transformations.html#sentence-transformations-pac
kage
[1028]: apidoc/textattack.transformations.sentence_transformations.html#module-textattack.transforma
tions.sentence_transformations.back_transcription
[1029]: apidoc/textattack.transformations.sentence_transformations.html#backtranscription-class
[1030]: apidoc/textattack.transformations.sentence_transformations.html#textattack.transformations.s
entence_transformations.back_transcription.BackTranscription
[1031]: apidoc/textattack.transformations.sentence_transformations.html#backtranslation-class
[1032]: apidoc/textattack.transformations.sentence_transformations.html#textattack.transformations.s
entence_transformations.back_translation.BackTranslation
[1033]: apidoc/textattack.transformations.sentence_transformations.html#sentencetransformation-class
[1034]: apidoc/textattack.transformations.sentence_transformations.html#textattack.transformations.s
entence_transformations.sentence_transformation.SentenceTransformation
[1035]: apidoc/textattack.transformations.word_insertions.html
[1036]: apidoc/textattack.transformations.word_insertions.html#word-insertions-package
[1037]: apidoc/textattack.transformations.word_insertions.html#module-textattack.transformations.wor
d_insertions.word_insertion
[1038]: apidoc/textattack.transformations.word_insertions.html#wordinsertion-class
[1039]: apidoc/textattack.transformations.word_insertions.html#textattack.transformations.word_inser
tions.word_insertion.WordInsertion
[1040]: apidoc/textattack.transformations.word_insertions.html#wordinsertionmaskedlm-class
[1041]: apidoc/textattack.transformations.word_insertions.html#textattack.transformations.word_inser
tions.word_insertion_masked_lm.WordInsertionMaskedLM
[1042]: apidoc/textattack.transformations.word_insertions.html#wordinsertionrandomsynonym-class
[1043]: apidoc/textattack.transformations.word_insertions.html#textattack.transformations.word_inser
tions.word_insertion_random_synonym.WordInsertionRandomSynonym
[1044]: apidoc/textattack.transformations.word_insertions.html#textattack.transformations.word_inser
tions.word_insertion_random_synonym.check_if_one_word
[1045]: apidoc/textattack.transformations.word_merges.html
[1046]: apidoc/textattack.transformations.word_merges.html#word-merges-package
[1047]: apidoc/textattack.transformations.word_merges.html#module-textattack.transformations.word_me
rges.word_merge
[1048]: apidoc/textattack.transformations.word_merges.html#word-merge
[1049]: apidoc/textattack.transformations.word_merges.html#textattack.transformations.word_merges.wo
rd_merge.WordMerge
[1050]: apidoc/textattack.transformations.word_merges.html#wordmergemaskedlm-class
[1051]: apidoc/textattack.transformations.word_merges.html#textattack.transformations.word_merges.wo
rd_merge_masked_lm.WordMergeMaskedLM
[1052]: apidoc/textattack.transformations.word_merges.html#textattack.transformations.word_merges.wo
rd_merge_masked_lm.find_merge_index
[1053]: apidoc/textattack.transformations.word_swaps.html
[1054]: apidoc/textattack.transformations.word_swaps.html#word-swaps-package
[1055]: apidoc/textattack.transformations.word_swaps.html#subpackages
[1056]: apidoc/textattack.transformations.word_swaps.chn_transformations.html
[1057]: apidoc/textattack.transformations.word_swaps.html#module-textattack.transformations.word_swa
ps.word_swap
[1058]: apidoc/textattack.transformations.word_swaps.html#word-swap
[1059]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap.WordSwap
[1060]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-changing-location
[1061]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_change_location.WordSwapChangeLocation
[1062]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_change_location.idx_to_words
[1063]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-changing-name
[1064]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_change_name.WordSwapChangeName
[1065]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-changing-number
[1066]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_change_number.WordSwapChangeNumber
[1067]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_change_number.idx_to_words
[1068]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-contraction
[1069]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_contract.WordSwapContract
[1070]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-invisible-deletions
[1071]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_deletions.WordSwapDeletions
[1072]: apidoc/textattack.transformations.word_swaps.html#word-swap-for-differential-evolution
[1073]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_differential_evolution.WordSwapDifferentialEvolution
[1074]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-embedding
[1075]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_embedding.WordSwapEmbedding
[1076]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_embedding.recover_word_case
[1077]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-extension
[1078]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_extend.WordSwapExtend
[1079]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-gradient
[1080]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_gradient_based.WordSwapGradientBased
[1081]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-homoglyph
[1082]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_homoglyph_swap.WordSwapHomoglyphSwap
[1083]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-openhownet
[1084]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_hownet.WordSwapHowNet
[1085]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_hownet.recover_word_case
[1086]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-inflections
[1087]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_inflections.WordSwapInflections
[1088]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-invisible-characters
[1089]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_invisible_characters.WordSwapInvisibleCharacters
[1090]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-bert-masked-lm
[1091]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_masked_lm.WordSwapMaskedLM
[1092]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_masked_lm.recover_word_case
[1093]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-neighboring-character-swap
[1094]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_neighboring_character_swap.WordSwapNeighboringCharacterSwap
[1095]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-swaps-characters-with-qwerty-
adjacent-keys
[1096]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_qwerty.WordSwapQWERTY
[1097]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-random-character-deletion
[1098]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_random_character_deletion.WordSwapRandomCharacterDeletion
[1099]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-random-character-insertion
[1100]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_random_character_insertion.WordSwapRandomCharacterInsertion
[1101]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-random-character-substitution
[1102]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_random_character_substitution.WordSwapRandomCharacterSubstitution
[1103]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-invisible-reorderings
[1104]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_reorderings.WordSwapReorderings
[1105]: apidoc/textattack.transformations.word_swaps.html#word-swap-by-swapping-synonyms-in-wordnet
[1106]: apidoc/textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word
_swap_wordnet.WordSwapWordNet
[1107]: apidoc/textattack.transformations.html#module-textattack.transformations.composite_transform
ation
[1108]: apidoc/textattack.transformations.html#composite-transformation
[1109]: apidoc/textattack.transformations.html#textattack.transformations.composite_transformation.C
ompositeTransformation
[1110]: apidoc/textattack.transformations.html#transformation-abstract-class
[1111]: apidoc/textattack.transformations.html#textattack.transformations.transformation.Transformat
ion
[1112]: apidoc/textattack.transformations.html#textattack.transformations.transformation.Transformat
ion.deterministic
[1113]: apidoc/textattack.transformations.html#word-deletion-transformation
[1114]: apidoc/textattack.transformations.html#textattack.transformations.word_deletion.WordDeletion
[1115]: apidoc/textattack.transformations.html#word-swap-transformation-by-swapping-the-order-of-wor
ds
[1116]: apidoc/textattack.transformations.html#textattack.transformations.word_innerswap_random.Word
InnerSwapRandom
[1117]: apidoc/textattack.transformations.html#textattack.transformations.word_innerswap_random.Word
InnerSwapRandom.deterministic
[1118]: apidoc/textattack.html#module-textattack.attack
[1119]: apidoc/textattack.html#attack-class
[1120]: apidoc/textattack.html#textattack.attack.Attack
[1121]: apidoc/textattack.html#textattack.attack.Attack.attack
[1122]: apidoc/textattack.html#textattack.attack.Attack.clear_cache
[1123]: apidoc/textattack.html#textattack.attack.Attack.cpu_
[1124]: apidoc/textattack.html#textattack.attack.Attack.cuda_
[1125]: apidoc/textattack.html#textattack.attack.Attack.filter_transformations
[1126]: apidoc/textattack.html#textattack.attack.Attack.get_indices_to_order
[1127]: apidoc/textattack.html#textattack.attack.Attack.get_transformations
[1128]: apidoc/textattack.html#attackargs-class
[1129]: apidoc/textattack.html#textattack.attack_args.AttackArgs
[1130]: apidoc/textattack.html#textattack.attack_args.AttackArgs.create_loggers_from_args
[1131]: apidoc/textattack.html#textattack.attack_args.AttackArgs.attack_n
[1132]: apidoc/textattack.html#textattack.attack_args.AttackArgs.checkpoint_dir
[1133]: apidoc/textattack.html#textattack.attack_args.AttackArgs.checkpoint_interval
[1134]: apidoc/textattack.html#textattack.attack_args.AttackArgs.csv_coloring_style
[1135]: apidoc/textattack.html#textattack.attack_args.AttackArgs.disable_stdout
[1136]: apidoc/textattack.html#textattack.attack_args.AttackArgs.enable_advance_metrics
[1137]: apidoc/textattack.html#textattack.attack_args.AttackArgs.log_summary_to_json
[1138]: apidoc/textattack.html#textattack.attack_args.AttackArgs.log_to_csv
[1139]: apidoc/textattack.html#textattack.attack_args.AttackArgs.log_to_txt
[1140]: apidoc/textattack.html#textattack.attack_args.AttackArgs.log_to_visdom
[1141]: apidoc/textattack.html#textattack.attack_args.AttackArgs.log_to_wandb
[1142]: apidoc/textattack.html#textattack.attack_args.AttackArgs.metrics
[1143]: apidoc/textattack.html#textattack.attack_args.AttackArgs.num_examples
[1144]: apidoc/textattack.html#textattack.attack_args.AttackArgs.num_examples_offset
[1145]: apidoc/textattack.html#textattack.attack_args.AttackArgs.num_successful_examples
[1146]: apidoc/textattack.html#textattack.attack_args.AttackArgs.num_workers_per_device
[1147]: apidoc/textattack.html#textattack.attack_args.AttackArgs.parallel
[1148]: apidoc/textattack.html#textattack.attack_args.AttackArgs.query_budget
[1149]: apidoc/textattack.html#textattack.attack_args.AttackArgs.random_seed
[1150]: apidoc/textattack.html#textattack.attack_args.AttackArgs.shuffle
[1151]: apidoc/textattack.html#textattack.attack_args.AttackArgs.silent
[1152]: apidoc/textattack.html#textattack.attack_args.CommandLineAttackArgs
[1153]: apidoc/textattack.html#attacker-class
[1154]: apidoc/textattack.html#textattack.attacker.Attacker
[1155]: apidoc/textattack.html#textattack.attacker.Attacker.attack_dataset
[1156]: apidoc/textattack.html#textattack.attacker.Attacker.attack_interactive
[1157]: apidoc/textattack.html#textattack.attacker.Attacker.from_checkpoint
[1158]: apidoc/textattack.html#textattack.attacker.Attacker.update_attack_args
[1159]: apidoc/textattack.html#textattack.attacker.attack_from_queue
[1160]: apidoc/textattack.html#textattack.attacker.pytorch_multiprocessing_workaround
[1161]: apidoc/textattack.html#textattack.attacker.set_env_variables
[1162]: apidoc/textattack.html#augmenterargs-class
[1163]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs
[1164]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.enable_advanced_metrics
[1165]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.exclude_original
[1166]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.fast_augment
[1167]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.high_yield
[1168]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.input_column
[1169]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.input_csv
[1170]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.interactive
[1171]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.output_csv
[1172]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.overwrite
[1173]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.pct_words_to_swap
[1174]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.random_seed
[1175]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.recipe
[1176]: apidoc/textattack.html#textattack.augment_args.AugmenterArgs.transformations_per_example
[1177]: apidoc/textattack.html#datasetargs-class
[1178]: apidoc/textattack.html#textattack.dataset_args.DatasetArgs
[1179]: apidoc/textattack.html#textattack.dataset_args.DatasetArgs.dataset_by_model
[1180]: apidoc/textattack.html#textattack.dataset_args.DatasetArgs.dataset_from_file
[1181]: apidoc/textattack.html#textattack.dataset_args.DatasetArgs.dataset_from_huggingface
[1182]: apidoc/textattack.html#textattack.dataset_args.DatasetArgs.dataset_split
[1183]: apidoc/textattack.html#textattack.dataset_args.DatasetArgs.filter_by_labels
[1184]: apidoc/textattack.html#modelargs-class
[1185]: apidoc/textattack.html#textattack.model_args.ModelArgs
[1186]: apidoc/textattack.html#textattack.model_args.ModelArgs.model
[1187]: apidoc/textattack.html#textattack.model_args.ModelArgs.model_from_file
[1188]: apidoc/textattack.html#textattack.model_args.ModelArgs.model_from_huggingface
[1189]: apidoc/textattack.html#trainer-class
[1190]: apidoc/textattack.html#textattack.trainer.Trainer
[1191]: apidoc/textattack.html#textattack.trainer.Trainer.evaluate
[1192]: apidoc/textattack.html#textattack.trainer.Trainer.evaluate_step
[1193]: apidoc/textattack.html#textattack.trainer.Trainer.get_eval_dataloader
[1194]: apidoc/textattack.html#textattack.trainer.Trainer.get_optimizer_and_scheduler
[1195]: apidoc/textattack.html#textattack.trainer.Trainer.get_train_dataloader
[1196]: apidoc/textattack.html#textattack.trainer.Trainer.train
[1197]: apidoc/textattack.html#textattack.trainer.Trainer.training_step
[1198]: apidoc/textattack.html#trainingargs-class
[1199]: apidoc/textattack.html#textattack.training_args.CommandLineTrainingArgs
[1200]: apidoc/textattack.html#textattack.training_args.CommandLineTrainingArgs.output_dir
[1201]: apidoc/textattack.html#textattack.training_args.TrainingArgs
[1202]: apidoc/textattack.html#textattack.training_args.TrainingArgs.alpha
[1203]: apidoc/textattack.html#textattack.training_args.TrainingArgs.attack_epoch_interval
[1204]: apidoc/textattack.html#textattack.training_args.TrainingArgs.attack_num_workers_per_device
[1205]: apidoc/textattack.html#textattack.training_args.TrainingArgs.checkpoint_interval_epochs
[1206]: apidoc/textattack.html#textattack.training_args.TrainingArgs.checkpoint_interval_steps
[1207]: apidoc/textattack.html#textattack.training_args.TrainingArgs.early_stopping_epochs
[1208]: apidoc/textattack.html#textattack.training_args.TrainingArgs.gradient_accumulation_steps
[1209]: apidoc/textattack.html#textattack.training_args.TrainingArgs.learning_rate
[1210]: apidoc/textattack.html#textattack.training_args.TrainingArgs.load_best_model_at_end
[1211]: apidoc/textattack.html#textattack.training_args.TrainingArgs.log_to_tb
[1212]: apidoc/textattack.html#textattack.training_args.TrainingArgs.log_to_wandb
[1213]: apidoc/textattack.html#textattack.training_args.TrainingArgs.logging_interval_step
[1214]: apidoc/textattack.html#textattack.training_args.TrainingArgs.num_clean_epochs
[1215]: apidoc/textattack.html#textattack.training_args.TrainingArgs.num_epochs
[1216]: apidoc/textattack.html#textattack.training_args.TrainingArgs.num_train_adv_examples
[1217]: apidoc/textattack.html#textattack.training_args.TrainingArgs.num_warmup_steps
[1218]: apidoc/textattack.html#textattack.training_args.TrainingArgs.output_dir
[1219]: apidoc/textattack.html#textattack.training_args.TrainingArgs.parallel
[1220]: apidoc/textattack.html#textattack.training_args.TrainingArgs.per_device_eval_batch_size
[1221]: apidoc/textattack.html#textattack.training_args.TrainingArgs.per_device_train_batch_size
[1222]: apidoc/textattack.html#textattack.training_args.TrainingArgs.query_budget_train
[1223]: apidoc/textattack.html#textattack.training_args.TrainingArgs.random_seed
[1224]: apidoc/textattack.html#textattack.training_args.TrainingArgs.save_last
[1225]: apidoc/textattack.html#textattack.training_args.TrainingArgs.tb_log_dir
[1226]: apidoc/textattack.html#textattack.training_args.TrainingArgs.wandb_project
[1227]: apidoc/textattack.html#textattack.training_args.TrainingArgs.weight_decay
[1228]: apidoc/textattack.html#textattack.training_args.default_output_dir
[1229]: 0_get_started/basic-Intro.html
[1230]: https://www.sphinx-doc.org/
[1231]: https://github.com/readthedocs/sphinx_rtd_theme
[1232]: https://readthedocs.org
