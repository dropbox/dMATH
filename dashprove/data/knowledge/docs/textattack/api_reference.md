* textattack package
* [ View page source][1]

# textattack package[][2]

Welcome to the API references for TextAttack!

What is TextAttack?

[TextAttack][3] is a Python framework for adversarial attacks, adversarial training, and data
augmentation in NLP.

TextAttack makes experimenting with the robustness of NLP models seamless, fast, and easy. It’s also
useful for NLP model training, adversarial training, and data augmentation.

TextAttack provides components for common NLP tasks like sentence encoding, grammar-checking, and
word replacement that can be used on their own.

## Subpackages[][4]

* [textattack.attack_recipes package][5]
  
  * [Attack Recipes Package:][6]
  * [Submodules][7]
    
    * [A2T (A2T: Attack for Adversarial Training Recipe)][8]
    * [`A2TYoo2021`][9]
      
      * [`A2TYoo2021.build()`][10]
    * [Attack Recipe Class][11]
    * [`AttackRecipe`][12]
      
      * [`AttackRecipe.build()`][13]
    * [Imperceptible Perturbations Algorithm][14]
    * [`BadCharacters2021`][15]
      
      * [`BadCharacters2021.build()`][16]
    * [BAE (BAE: BERT-Based Adversarial Examples)][17]
    * [`BAEGarg2019`][18]
      
      * [`BAEGarg2019.build()`][19]
    * [BERT-Attack:][20]
    * [`BERTAttackLi2020`][21]
      
      * [`BERTAttackLi2020.build()`][22]
    * [CheckList:][23]
    * [`CheckList2020`][24]
      
      * [`CheckList2020.build()`][25]
    * [Attack Chinese Recipe][26]
    * [`ChineseRecipe`][27]
      
      * [`ChineseRecipe.build()`][28]
    * [CLARE Recipe][29]
    * [`CLARE2020`][30]
      
      * [`CLARE2020.build()`][31]
    * [DeepWordBug][32]
    * [`DeepWordBugGao2018`][33]
      
      * [`DeepWordBugGao2018.build()`][34]
    * [Faster Alzantot Genetic Algorithm][35]
    * [`FasterGeneticAlgorithmJia2019`][36]
      
      * [`FasterGeneticAlgorithmJia2019.build()`][37]
    * [Attack French Recipe][38]
    * [`FrenchRecipe`][39]
      
      * [`FrenchRecipe.build()`][40]
    * [Alzantot Genetic Algorithm][41]
    * [`GeneticAlgorithmAlzantot2018`][42]
      
      * [`GeneticAlgorithmAlzantot2018.build()`][43]
    * [HotFlip][44]
    * [`HotFlipEbrahimi2017`][45]
      
      * [`HotFlipEbrahimi2017.build()`][46]
    * [Improved Genetic Algorithm][47]
    * [`IGAWang2019`][48]
      
      * [`IGAWang2019.build()`][49]
    * [Input Reduction][50]
    * [`InputReductionFeng2018`][51]
      
      * [`InputReductionFeng2018.build()`][52]
    * [Kuleshov2017][53]
    * [`Kuleshov2017`][54]
      
      * [`Kuleshov2017.build()`][55]
    * [MORPHEUS2020][56]
    * [`MorpheusTan2020`][57]
      
      * [`MorpheusTan2020.build()`][58]
    * [Pruthi2019: Combating with Robust Word Recognition][59]
    * [`Pruthi2019`][60]
      
      * [`Pruthi2019.build()`][61]
    * [Particle Swarm Optimization][62]
    * [`PSOZang2020`][63]
      
      * [`PSOZang2020.build()`][64]
    * [PWWS][65]
    * [`PWWSRen2019`][66]
      
      * [`PWWSRen2019.build()`][67]
    * [Seq2Sick][68]
    * [`Seq2SickCheng2018BlackBox`][69]
      
      * [`Seq2SickCheng2018BlackBox.build()`][70]
    * [Attack Spanish Recipe][71]
    * [`SpanishRecipe`][72]
      
      * [`SpanishRecipe.build()`][73]
    * [TextBugger][74]
    * [`TextBuggerLi2018`][75]
      
      * [`TextBuggerLi2018.build()`][76]
    * [TextFooler (Is BERT Really Robust?)][77]
    * [`TextFoolerJin2019`][78]
      
      * [`TextFoolerJin2019.build()`][79]
* [textattack.attack_results package][80]
  
  * [TextAttack attack_results Package][81]
  * [Submodules][82]
    
    * [AttackResult Class][83]
    * [`AttackResult`][84]
      
      * [`AttackResult.diff_color()`][85]
      * [`AttackResult.goal_function_result_str()`][86]
      * [`AttackResult.original_text()`][87]
      * [`AttackResult.perturbed_text()`][88]
      * [`AttackResult.str_lines()`][89]
    * [FailedAttackResult Class][90]
    * [`FailedAttackResult`][91]
      
      * [`FailedAttackResult.goal_function_result_str()`][92]
      * [`FailedAttackResult.str_lines()`][93]
    * [MaximizedAttackResult Class][94]
    * [`MaximizedAttackResult`][95]
    * [SkippedAttackResult Class][96]
    * [`SkippedAttackResult`][97]
      
      * [`SkippedAttackResult.goal_function_result_str()`][98]
      * [`SkippedAttackResult.str_lines()`][99]
    * [SuccessfulAttackResult Class][100]
    * [`SuccessfulAttackResult`][101]
* [textattack.augmentation package][102]
  
  * [TextAttack augmentation package:][103]
  * [Submodules][104]
    
    * [Augmenter Class][105]
    * [`AugmentationResult`][106]
      
      * [`AugmentationResult.tempResult`][107]
    * [`Augmenter`][108]
      
      * [`Augmenter.augment()`][109]
      * [`Augmenter.augment_many()`][110]
      * [`Augmenter.augment_text_with_ids()`][111]
    * [Augmenter Recipes:][112]
    * [`BackTranscriptionAugmenter`][113]
    * [`BackTranslationAugmenter`][114]
    * [`CLAREAugmenter`][115]
    * [`CharSwapAugmenter`][116]
    * [`CheckListAugmenter`][117]
    * [`DeletionAugmenter`][118]
    * [`EasyDataAugmenter`][119]
      
      * [`EasyDataAugmenter.augment()`][120]
    * [`EmbeddingAugmenter`][121]
    * [`SwapAugmenter`][122]
    * [`SynonymInsertionAugmenter`][123]
    * [`WordNetAugmenter`][124]
* [textattack.commands package][125]
  
  * [TextAttack commands Package][126]
  * [Submodules][127]
    
    * [AttackCommand class][128]
    * [`AttackCommand`][129]
      
      * [`AttackCommand.register_subcommand()`][130]
      * [`AttackCommand.run()`][131]
    * [AttackResumeCommand class][132]
    * [`AttackResumeCommand`][133]
      
      * [`AttackResumeCommand.register_subcommand()`][134]
      * [`AttackResumeCommand.run()`][135]
    * [AugmentCommand class][136]
    * [`AugmentCommand`][137]
      
      * [`AugmentCommand.register_subcommand()`][138]
      * [`AugmentCommand.run()`][139]
    * [BenchmarkRecipeCommand class][140]
    * [`BenchmarkRecipeCommand`][141]
      
      * [`BenchmarkRecipeCommand.register_subcommand()`][142]
      * [`BenchmarkRecipeCommand.run()`][143]
    * [EvalModelCommand class][144]
    * [`EvalModelCommand`][145]
      
      * [`EvalModelCommand.get_preds()`][146]
      * [`EvalModelCommand.register_subcommand()`][147]
      * [`EvalModelCommand.run()`][148]
      * [`EvalModelCommand.test_model_on_dataset()`][149]
    * [`ModelEvalArgs`][150]
      
      * [`ModelEvalArgs.batch_size`][151]
      * [`ModelEvalArgs.num_examples`][152]
      * [`ModelEvalArgs.num_examples_offset`][153]
      * [`ModelEvalArgs.random_seed`][154]
    * [ListThingsCommand class][155]
    * [`ListThingsCommand`][156]
      
      * [`ListThingsCommand.register_subcommand()`][157]
      * [`ListThingsCommand.run()`][158]
      * [`ListThingsCommand.things()`][159]
    * [PeekDatasetCommand class][160]
    * [`PeekDatasetCommand`][161]
      
      * [`PeekDatasetCommand.register_subcommand()`][162]
      * [`PeekDatasetCommand.run()`][163]
    * [TextAttack CLI main class][164]
    * [`main()`][165]
    * [`TextAttackCommand`][166]
      
      * [`TextAttackCommand.register_subcommand()`][167]
      * [`TextAttackCommand.run()`][168]
    * [TrainModelCommand class][169]
    * [`TrainModelCommand`][170]
      
      * [`TrainModelCommand.register_subcommand()`][171]
      * [`TrainModelCommand.run()`][172]
* [textattack.constraints package][173]
  
  * [Constraints][174]
  * [Subpackages][175]
    
    * [textattack.constraints.grammaticality package][176]
      
      * [Grammaticality:][177]
      * [Subpackages][178]
        
        * [textattack.constraints.grammaticality.language_models package][179]
          
          * [non-pre Language Models:][180]
          * [Subpackages][181]
          * [Submodules][182]
      * [Submodules][183]
        
        * [CoLA for Grammaticality][184]
        * [`COLA`][185]
          
          * [`COLA.clear_cache()`][186]
          * [`COLA.extra_repr_keys()`][187]
        * [LanguageTool Grammar Checker][188]
        * [`LanguageTool`][189]
          
          * [`LanguageTool.extra_repr_keys()`][190]
          * [`LanguageTool.get_errors()`][191]
        * [Part of Speech Constraint][192]
        * [`PartOfSpeech`][193]
          
          * [`PartOfSpeech.check_compatibility()`][194]
          * [`PartOfSpeech.clear_cache()`][195]
          * [`PartOfSpeech.extra_repr_keys()`][196]
    * [textattack.constraints.overlap package][197]
      
      * [Overlap Constraints][198]
      * [Submodules][199]
        
        * [BLEU Constraints][200]
        * [`BLEU`][201]
          
          * [`BLEU.extra_repr_keys()`][202]
        * [chrF Constraints][203]
        * [`chrF`][204]
          
          * [`chrF.extra_repr_keys()`][205]
        * [Edit Distance Constraints][206]
        * [`LevenshteinEditDistance`][207]
          
          * [`LevenshteinEditDistance.extra_repr_keys()`][208]
        * [Max Perturb Words Constraints][209]
        * [`MaxWordsPerturbed`][210]
          
          * [`MaxWordsPerturbed.extra_repr_keys()`][211]
        * [METEOR Constraints][212]
        * [`METEOR`][213]
          
          * [`METEOR.extra_repr_keys()`][214]
    * [textattack.constraints.pre_transformation package][215]
      
      * [Pre-Transformation:][216]
      * [Submodules][217]
        
        * [Input Column Modification][218]
        * [`InputColumnModification`][219]
          
          * [`InputColumnModification.extra_repr_keys()`][220]
        * [Max Modification Rate][221]
        * [`MaxModificationRate`][222]
          
          * [`MaxModificationRate.extra_repr_keys()`][223]
        * [Max Modification Rate][224]
        * [`MaxNumWordsModified`][225]
          
          * [`MaxNumWordsModified.extra_repr_keys()`][226]
        * [Max Word Index Modification][227]
        * [`MaxWordIndexModification`][228]
          
          * [`MaxWordIndexModification.extra_repr_keys()`][229]
        * [Min Word Lenth][230]
        * [`MinWordLength`][231]
        * [Repeat Modification][232]
        * [`RepeatModification`][233]
        * [Stopword Modification][234]
        * [`StopwordModification`][235]
          
          * [`StopwordModification.check_compatibility()`][236]
        * [`UnmodifiableIndices`][237]
          
          * [`UnmodifiableIndices.extra_repr_keys()`][238]
        * [`UnmodifablePhrases`][239]
          
          * [`UnmodifablePhrases.extra_repr_keys()`][240]
    * [textattack.constraints.semantics package][241]
      
      * [Semantic Constraints][242]
      * [Subpackages][243]
        
        * [textattack.constraints.semantics.sentence_encoders package][244]
          
          * [Sentence Encoder Constraint][245]
          * [Subpackages][246]
          * [Submodules][247]
      * [Submodules][248]
        
        * [BERT Score][249]
        * [`BERTScore`][250]
          
          * [`BERTScore.extra_repr_keys()`][251]
          * [`BERTScore.SCORE_TYPE2IDX`][252]
        * [Word Embedding Distance][253]
        * [`WordEmbeddingDistance`][254]
          
          * [`WordEmbeddingDistance.check_compatibility()`][255]
          * [`WordEmbeddingDistance.extra_repr_keys()`][256]
          * [`WordEmbeddingDistance.get_cos_sim()`][257]
          * [`WordEmbeddingDistance.get_mse_dist()`][258]
  * [Submodules][259]
    
    * [TextAttack Constraint Class][260]
    * [`Constraint`][261]
      
      * [`Constraint.call_many()`][262]
      * [`Constraint.check_compatibility()`][263]
      * [`Constraint.extra_repr_keys()`][264]
    * [Pre-Transformation Constraint Class][265]
    * [`PreTransformationConstraint`][266]
      
      * [`PreTransformationConstraint.check_compatibility()`][267]
      * [`PreTransformationConstraint.extra_repr_keys()`][268]
* [textattack.datasets package][269]
  
  * [datasets package:][270]
  * [Subpackages][271]
    
    * [textattack.datasets.helpers package][272]
      
      * [Dataset Helpers][273]
      * [Submodules][274]
        
        * [Ted Multi TranslationDataset Class][275]
        * [`TedMultiTranslationDataset`][276]
  * [Submodules][277]
    
    * [Dataset Class][278]
    * [`Dataset`][279]
      
      * [`Dataset.filter_by_labels_()`][280]
      * [`Dataset.shuffle()`][281]
    * [HuggingFaceDataset Class][282]
    * [`HuggingFaceDataset`][283]
      
      * [`HuggingFaceDataset.filter_by_labels_()`][284]
      * [`HuggingFaceDataset.shuffle()`][285]
    * [`get_datasets_dataset_columns()`][286]
* [textattack.goal_function_results package][287]
  
  * [Goal Function Result package:][288]
  * [Subpackages][289]
    
    * [textattack.goal_function_results.custom package][290]
      
      * [Custom Goal Function Result package:][291]
      * [Submodules][292]
        
        * [LogitSumGoalFunctionResult Class][293]
        * [`LogitSumGoalFunctionResult`][294]
          
          * [`LogitSumGoalFunctionResult.get_colored_output()`][295]
          * [`LogitSumGoalFunctionResult.get_text_color_input()`][296]
          * [`LogitSumGoalFunctionResult.get_text_color_perturbed()`][297]
        * [NamedEntityRecognitionoalFunctionResult Class][298]
        * [`NamedEntityRecognitionGoalFunctionResult`][299]
          
          * [`NamedEntityRecognitionGoalFunctionResult.get_colored_output()`][300]
          * [`NamedEntityRecognitionGoalFunctionResult.get_text_color_input()`][301]
          * [`NamedEntityRecognitionGoalFunctionResult.get_text_color_perturbed()`][302]
        * [TargetedBonusGoalFunctionResult Class][303]
        * [`TargetedBonusGoalFunctionResult`][304]
          
          * [`TargetedBonusGoalFunctionResult.get_colored_output()`][305]
          * [`TargetedBonusGoalFunctionResult.get_text_color_input()`][306]
          * [`TargetedBonusGoalFunctionResult.get_text_color_perturbed()`][307]
        * [TargetedStrictGoalFunctionResult Class][308]
        * [`TargetedStrictGoalFunctionResult`][309]
          
          * [`TargetedStrictGoalFunctionResult.get_colored_output()`][310]
          * [`TargetedStrictGoalFunctionResult.get_text_color_input()`][311]
          * [`TargetedStrictGoalFunctionResult.get_text_color_perturbed()`][312]
  * [Submodules][313]
    
    * [ClassificationGoalFunctionResult Class][314]
    * [`ClassificationGoalFunctionResult`][315]
      
      * [`ClassificationGoalFunctionResult.get_colored_output()`][316]
      * [`ClassificationGoalFunctionResult.get_text_color_input()`][317]
      * [`ClassificationGoalFunctionResult.get_text_color_perturbed()`][318]
    * [GoalFunctionResult class][319]
    * [`GoalFunctionResult`][320]
      
      * [`GoalFunctionResult.get_colored_output()`][321]
      * [`GoalFunctionResult.get_text_color_input()`][322]
      * [`GoalFunctionResult.get_text_color_perturbed()`][323]
    * [`GoalFunctionResultStatus`][324]
      
      * [`GoalFunctionResultStatus.MAXIMIZING`][325]
      * [`GoalFunctionResultStatus.SEARCHING`][326]
      * [`GoalFunctionResultStatus.SKIPPED`][327]
      * [`GoalFunctionResultStatus.SUCCEEDED`][328]
    * [TextToTextGoalFunctionResult Class][329]
    * [`TextToTextGoalFunctionResult`][330]
      
      * [`TextToTextGoalFunctionResult.get_colored_output()`][331]
      * [`TextToTextGoalFunctionResult.get_text_color_input()`][332]
      * [`TextToTextGoalFunctionResult.get_text_color_perturbed()`][333]
* [textattack.goal_functions package][334]
  
  * [Goal Functions][335]
  * [Subpackages][336]
    
    * [textattack.goal_functions.classification package][337]
      
      * [Goal fucntion for Classification][338]
      * [Submodules][339]
        
        * [Determine for if an attack has been successful in Classification][340]
        * [`ClassificationGoalFunction`][341]
          
          * [`ClassificationGoalFunction.extra_repr_keys()`][342]
        * [Determine if an attack has been successful in Hard Label Classficiation.][343]
        * [`HardLabelClassification`][344]
        * [Determine if maintaining the same predicted label (input reduction)][345]
        * [`InputReduction`][346]
          
          * [`InputReduction.extra_repr_keys()`][347]
        * [Determine if an attack has been successful in targeted Classification][348]
        * [`TargetedClassification`][349]
          
          * [`TargetedClassification.extra_repr_keys()`][350]
        * [Determine successful in untargeted Classification][351]
        * [`UntargetedClassification`][352]
    * [textattack.goal_functions.custom package][353]
      
      * [Custom goal functions][354]
      * [Submodules][355]
        
        * [Goal Function for Logit sum][356]
        * [`LogitSum`][357]
        * [Goal Function for NamedEntityRecognition][358]
        * [`NamedEntityRecognition`][359]
        * [Goal Function for Targeted classification with bonus score][360]
        * [`TargetedBonus`][361]
        * [Goal Function for Strict targeted classification][362]
        * [`TargetedStrict`][363]
    * [textattack.goal_functions.text package][364]
      
      * [Goal Function for Text to Text case][365]
      * [Submodules][366]
        
        * [`MaximizeLevenshtein`][367]
          
          * [`MaximizeLevenshtein.clear_cache()`][368]
          * [`MaximizeLevenshtein.extra_repr_keys()`][369]
        * [Goal Function for Attempts to minimize the BLEU score][370]
        * [`MinimizeBleu`][371]
          
          * [`MinimizeBleu.clear_cache()`][372]
          * [`MinimizeBleu.extra_repr_keys()`][373]
          * [`MinimizeBleu.EPS`][374]
        * [`get_bleu()`][375]
        * [Goal Function for seq2sick][376]
        * [`NonOverlappingOutput`][377]
          
          * [`NonOverlappingOutput.clear_cache()`][378]
        * [`get_words_cached()`][379]
        * [`word_difference_score()`][380]
        * [Goal Function for TextToText][381]
        * [`TextToTextGoalFunction`][382]
  * [Submodules][383]
    
    * [GoalFunction Class][384]
    * [`GoalFunction`][385]
      
      * [`GoalFunction.clear_cache()`][386]
      * [`GoalFunction.extra_repr_keys()`][387]
      * [`GoalFunction.get_output()`][388]
      * [`GoalFunction.get_result()`][389]
      * [`GoalFunction.get_results()`][390]
      * [`GoalFunction.init_attack_example()`][391]
* [textattack.llms package][392]
  
  * [Large Language Models][393]
  * [Submodules][394]
    
    * [`ChatGptWrapper`][395]
    * [`HuggingFaceLLMWrapper`][396]
* [textattack.loggers package][397]
  
  * [Misc Loggers: Loggers track, visualize, and export attack results.][398]
  * [Submodules][399]
    
    * [Managing Attack Logs.][400]
    * [`AttackLogManager`][401]
      
      * [`AttackLogManager.add_output_csv()`][402]
      * [`AttackLogManager.add_output_file()`][403]
      * [`AttackLogManager.add_output_summary_json()`][404]
      * [`AttackLogManager.disable_color()`][405]
      * [`AttackLogManager.enable_stdout()`][406]
      * [`AttackLogManager.enable_visdom()`][407]
      * [`AttackLogManager.enable_wandb()`][408]
      * [`AttackLogManager.flush()`][409]
      * [`AttackLogManager.log_attack_details()`][410]
      * [`AttackLogManager.log_result()`][411]
      * [`AttackLogManager.log_results()`][412]
      * [`AttackLogManager.log_sep()`][413]
      * [`AttackLogManager.log_summary()`][414]
      * [`AttackLogManager.log_summary_rows()`][415]
      * [`AttackLogManager.metrics`][416]
    * [Attack Logs to CSV][417]
    * [`CSVLogger`][418]
      
      * [`CSVLogger.close()`][419]
      * [`CSVLogger.flush()`][420]
      * [`CSVLogger.log_attack_result()`][421]
    * [Attack Logs to file][422]
    * [`FileLogger`][423]
      
      * [`FileLogger.close()`][424]
      * [`FileLogger.flush()`][425]
      * [`FileLogger.log_attack_result()`][426]
      * [`FileLogger.log_sep()`][427]
      * [`FileLogger.log_summary_rows()`][428]
    * [Attack Summary Results Logs to Json][429]
    * [`JsonSummaryLogger`][430]
      
      * [`JsonSummaryLogger.close()`][431]
      * [`JsonSummaryLogger.flush()`][432]
      * [`JsonSummaryLogger.log_summary_rows()`][433]
    * [Attack Logger Wrapper][434]
    * [`Logger`][435]
      
      * [`Logger.close()`][436]
      * [`Logger.flush()`][437]
      * [`Logger.log_attack_result()`][438]
      * [`Logger.log_hist()`][439]
      * [`Logger.log_sep()`][440]
      * [`Logger.log_summary_rows()`][441]
    * [Attack Logs to Visdom][442]
    * [`VisdomLogger`][443]
      
      * [`VisdomLogger.bar()`][444]
      * [`VisdomLogger.flush()`][445]
      * [`VisdomLogger.hist()`][446]
      * [`VisdomLogger.log_attack_result()`][447]
      * [`VisdomLogger.log_hist()`][448]
      * [`VisdomLogger.log_summary_rows()`][449]
      * [`VisdomLogger.table()`][450]
      * [`VisdomLogger.text()`][451]
    * [`port_is_open()`][452]
    * [Attack Logs to WandB][453]
    * [`WeightsAndBiasesLogger`][454]
      
      * [`WeightsAndBiasesLogger.log_attack_result()`][455]
      * [`WeightsAndBiasesLogger.log_sep()`][456]
      * [`WeightsAndBiasesLogger.log_summary_rows()`][457]
* [textattack.metrics package][458]
  
  * [metrics package: to calculate advanced metrics for evaluting attacks and augmented text][459]
  * [Subpackages][460]
    
    * [textattack.metrics.attack_metrics package][461]
      
      * [attack_metrics package:][462]
      * [Submodules][463]
        
        * [Metrics on AttackQueries][464]
        * [`AttackQueries`][465]
          
          * [`AttackQueries.avg_num_queries()`][466]
          * [`AttackQueries.calculate()`][467]
        * [Metrics on AttackSuccessRate][468]
        * [`AttackSuccessRate`][469]
          
          * [`AttackSuccessRate.attack_accuracy_perc()`][470]
          * [`AttackSuccessRate.attack_success_rate_perc()`][471]
          * [`AttackSuccessRate.calculate()`][472]
          * [`AttackSuccessRate.original_accuracy_perc()`][473]
        * [Metrics on perturbed words][474]
        * [`WordsPerturbed`][475]
          
          * [`WordsPerturbed.avg_number_word_perturbed_num()`][476]
          * [`WordsPerturbed.avg_perturbation_perc()`][477]
          * [`WordsPerturbed.calculate()`][478]
    * [textattack.metrics.quality_metrics package][479]
      
      * [Metrics on Quality package][480]
      * [Submodules][481]
        
        * [BERTScoreMetric class:][482]
        * [`BERTScoreMetric`][483]
          
          * [`BERTScoreMetric.calculate()`][484]
        * [MeteorMetric class:][485]
        * [`MeteorMetric`][486]
          
          * [`MeteorMetric.calculate()`][487]
        * [Perplexity Metric:][488]
        * [`Perplexity`][489]
          
          * [`Perplexity.calc_ppl()`][490]
          * [`Perplexity.calculate()`][491]
        * [USEMetric class:][492]
        * [`SBERTMetric`][493]
          
          * [`SBERTMetric.calculate()`][494]
        * [USEMetric class:][495]
        * [`USEMetric`][496]
          
          * [`USEMetric.calculate()`][497]
  * [Submodules][498]
    
    * [Metric Class][499]
    * [`Metric`][500]
      
      * [`Metric.calculate()`][501]
    * [Attack Metric Quality Recipes:][502]
    * [`AdvancedAttackMetric`][503]
      
      * [`AdvancedAttackMetric.calculate()`][504]
* [textattack.models package][505]
  
  * [Models][506]
    
    * [Models User-specified][507]
    * [Models Pre-trained][508]
    * [Model Wrappers][509]
  * [Subpackages][510]
    
    * [textattack.models.helpers package][511]
      
      * [Moderl Helpers][512]
      * [Submodules][513]
        
        * [Glove Embedding][514]
        * [`EmbeddingLayer`][515]
          
          * [`EmbeddingLayer.forward()`][516]
          * [`EmbeddingLayer.training`][517]
        * [`GloveEmbeddingLayer`][518]
          
          * [`GloveEmbeddingLayer.EMBEDDING_PATH`][519]
          * [`GloveEmbeddingLayer.training`][520]
        * [LSTM 4 Classification][521]
        * [`LSTMForClassification`][522]
          
          * [`LSTMForClassification.forward()`][523]
          * [`LSTMForClassification.from_pretrained()`][524]
          * [`LSTMForClassification.get_input_embeddings()`][525]
          * [`LSTMForClassification.load_from_disk()`][526]
          * [`LSTMForClassification.save_pretrained()`][527]
          * [`LSTMForClassification.training`][528]
        * [T5 model trained to generate text from text][529]
        * [`T5ForTextToText`][530]
          
          * [`T5ForTextToText.from_pretrained()`][531]
          * [`T5ForTextToText.get_input_embeddings()`][532]
          * [`T5ForTextToText.save_pretrained()`][533]
          * [`T5ForTextToText.training`][534]
        * [Util function for Model Wrapper][535]
        * [`load_cached_state_dict()`][536]
        * [Word CNN for Classification][537]
        * [`CNNTextLayer`][538]
          
          * [`CNNTextLayer.forward()`][539]
          * [`CNNTextLayer.training`][540]
        * [`WordCNNForClassification`][541]
          
          * [`WordCNNForClassification.forward()`][542]
          * [`WordCNNForClassification.from_pretrained()`][543]
          * [`WordCNNForClassification.get_input_embeddings()`][544]
          * [`WordCNNForClassification.load_from_disk()`][545]
          * [`WordCNNForClassification.save_pretrained()`][546]
          * [`WordCNNForClassification.training`][547]
    * [textattack.models.tokenizers package][548]
      
      * [Tokenizers for Model Wrapper][549]
      * [Submodules][550]
        
        * [Glove Tokenizer][551]
        * [`GloveTokenizer`][552]
          
          * [`GloveTokenizer.batch_encode()`][553]
          * [`GloveTokenizer.convert_ids_to_tokens()`][554]
          * [`GloveTokenizer.encode()`][555]
        * [`WordLevelTokenizer`][556]
        * [T5 Tokenizer][557]
        * [`T5Tokenizer`][558]
          
          * [`T5Tokenizer.decode()`][559]
    * [textattack.models.wrappers package][560]
      
      * [Model Wrappers Package][561]
      * [Submodules][562]
        
        * [HuggingFace Model Wrapper][563]
        * [`HuggingFaceModelWrapper`][564]
          
          * [`HuggingFaceModelWrapper.get_grad()`][565]
        * [ModelWrapper class][566]
        * [`ModelWrapper`][567]
          
          * [`ModelWrapper.get_grad()`][568]
          * [`ModelWrapper.tokenize()`][569]
        * [PyTorch Model Wrapper][570]
        * [`PyTorchModelWrapper`][571]
          
          * [`PyTorchModelWrapper.get_grad()`][572]
          * [`PyTorchModelWrapper.to()`][573]
        * [scikit-learn Model Wrapper][574]
        * [`SklearnModelWrapper`][575]
          
          * [`SklearnModelWrapper.get_grad()`][576]
        * [TensorFlow Model Wrapper][577]
        * [`TensorFlowModelWrapper`][578]
          
          * [`TensorFlowModelWrapper.get_grad()`][579]
* [textattack.prompt_augmentation package][580]
  
  * [Prompt Augmentation][581]
  * [Submodules][582]
    
    * [`PromptAugmentationPipeline`][583]
* [textattack.search_methods package][584]
  
  * [Search Methods][585]
  * [Submodules][586]
    
    * [Reimplementation of search method from Generating Natural Language Adversarial Examples][587]
    * [`AlzantotGeneticAlgorithm`][588]
    * [Beam Search][589]
    * [`BeamSearch`][590]
      
      * [`BeamSearch.extra_repr_keys()`][591]
      * [`BeamSearch.perform_search()`][592]
      * [`BeamSearch.is_black_box`][593]
    * [`DifferentialEvolution`][594]
      
      * [`DifferentialEvolution.check_transformation_compatibility()`][595]
      * [`DifferentialEvolution.extra_repr_keys()`][596]
      * [`DifferentialEvolution.perform_search()`][597]
      * [`DifferentialEvolution.is_black_box`][598]
    * [Genetic Algorithm Word Swap][599]
    * [`GeneticAlgorithm`][600]
      
      * [`GeneticAlgorithm.check_transformation_compatibility()`][601]
      * [`GeneticAlgorithm.extra_repr_keys()`][602]
      * [`GeneticAlgorithm.perform_search()`][603]
      * [`GeneticAlgorithm.is_black_box`][604]
    * [Greedy Search][605]
    * [`GreedySearch`][606]
      
      * [`GreedySearch.extra_repr_keys()`][607]
    * [Greedy Word Swap with Word Importance Ranking][608]
    * [`GreedyWordSwapWIR`][609]
      
      * [`GreedyWordSwapWIR.check_transformation_compatibility()`][610]
      * [`GreedyWordSwapWIR.extra_repr_keys()`][611]
      * [`GreedyWordSwapWIR.perform_search()`][612]
      * [`GreedyWordSwapWIR.is_black_box`][613]
    * [Reimplementation of search method from Xiaosen Wang, Hao Jin, Kun He (2019).][614]
    * [`ImprovedGeneticAlgorithm`][615]
      
      * [`ImprovedGeneticAlgorithm.extra_repr_keys()`][616]
    * [Particle Swarm Optimization][617]
    * [`ParticleSwarmOptimization`][618]
      
      * [`ParticleSwarmOptimization.check_transformation_compatibility()`][619]
      * [`ParticleSwarmOptimization.extra_repr_keys()`][620]
      * [`ParticleSwarmOptimization.perform_search()`][621]
      * [`ParticleSwarmOptimization.is_black_box`][622]
    * [`normalize()`][623]
    * [Population based Search abstract class][624]
    * [`PopulationBasedSearch`][625]
    * [`PopulationMember`][626]
      
      * [`PopulationMember.num_words`][627]
      * [`PopulationMember.score`][628]
      * [`PopulationMember.words`][629]
    * [Search Method Abstract Class][630]
    * [`SearchMethod`][631]
      
      * [`SearchMethod.check_transformation_compatibility()`][632]
      * [`SearchMethod.get_victim_model()`][633]
      * [`SearchMethod.perform_search()`][634]
      * [`SearchMethod.is_black_box`][635]
* [textattack.shared package][636]
  
  * [Shared TextAttack Functions][637]
  * [Subpackages][638]
    
    * [textattack.shared.utils package][639]
      
      * [Submodules][640]
        
        * [`LazyLoader`][641]
        * [`load_module_from_file()`][642]
        * [`download_from_s3()`][643]
        * [`download_from_url()`][644]
        * [`http_get()`][645]
        * [`path_in_cache()`][646]
        * [`s3_url()`][647]
        * [`set_cache_dir()`][648]
        * [`unzip_file()`][649]
        * [`get_textattack_model_num_labels()`][650]
        * [`hashable()`][651]
        * [`html_style_from_dict()`][652]
        * [`html_table_from_rows()`][653]
        * [`load_textattack_model_from_path()`][654]
        * [`set_seed()`][655]
        * [`sigmoid()`][656]
        * [`ANSI_ESCAPE_CODES`][657]
          
          * [`ANSI_ESCAPE_CODES.BOLD`][658]
          * [`ANSI_ESCAPE_CODES.BROWN`][659]
          * [`ANSI_ESCAPE_CODES.CYAN`][660]
          * [`ANSI_ESCAPE_CODES.FAIL`][661]
          * [`ANSI_ESCAPE_CODES.GRAY`][662]
          * [`ANSI_ESCAPE_CODES.HEADER`][663]
          * [`ANSI_ESCAPE_CODES.OKBLUE`][664]
          * [`ANSI_ESCAPE_CODES.OKGREEN`][665]
          * [`ANSI_ESCAPE_CODES.ORANGE`][666]
          * [`ANSI_ESCAPE_CODES.PINK`][667]
          * [`ANSI_ESCAPE_CODES.PURPLE`][668]
          * [`ANSI_ESCAPE_CODES.STOP`][669]
          * [`ANSI_ESCAPE_CODES.UNDERLINE`][670]
          * [`ANSI_ESCAPE_CODES.WARNING`][671]
          * [`ANSI_ESCAPE_CODES.YELLOW`][672]
        * [`ReprMixin`][673]
          
          * [`ReprMixin.extra_repr_keys()`][674]
        * [`TextAttackFlairTokenizer`][675]
          
          * [`TextAttackFlairTokenizer.tokenize()`][676]
        * [`add_indent()`][677]
        * [`check_if_punctuations()`][678]
        * [`check_if_subword()`][679]
        * [`color_from_label()`][680]
        * [`color_from_output()`][681]
        * [`color_text()`][682]
        * [`default_class_repr()`][683]
        * [`flair_tag()`][684]
        * [`has_letter()`][685]
        * [`is_one_word()`][686]
        * [`process_label_name()`][687]
        * [`strip_BPE_artifacts()`][688]
        * [`words_from_text()`][689]
        * [`zip_flair_result()`][690]
        * [`zip_stanza_result()`][691]
        * [`batch_model_predict()`][692]
  * [Submodules][693]
    
    * [Attacked Text Class][694]
    * [`AttackedText`][695]
      
      * [`AttackedText.align_with_model_tokens()`][696]
      * [`AttackedText.all_words_diff()`][697]
      * [`AttackedText.convert_from_original_idxs()`][698]
      * [`AttackedText.delete_word_at_index()`][699]
      * [`AttackedText.first_word_diff()`][700]
      * [`AttackedText.first_word_diff_index()`][701]
      * [`AttackedText.free_memory()`][702]
      * [`AttackedText.generate_new_attacked_text()`][703]
      * [`AttackedText.get_deletion_indices()`][704]
      * [`AttackedText.insert_text_after_word_index()`][705]
      * [`AttackedText.insert_text_before_word_index()`][706]
      * [`AttackedText.ith_word_diff()`][707]
      * [`AttackedText.ner_of_word_index()`][708]
      * [`AttackedText.pos_of_word_index()`][709]
      * [`AttackedText.printable_text()`][710]
      * [`AttackedText.replace_word_at_index()`][711]
      * [`AttackedText.replace_words_at_indices()`][712]
      * [`AttackedText.text_after_word_index()`][713]
      * [`AttackedText.text_until_word_index()`][714]
      * [`AttackedText.text_window_around_index()`][715]
      * [`AttackedText.words_diff_num()`][716]
      * [`AttackedText.words_diff_ratio()`][717]
      * [`AttackedText.SPLIT_TOKEN`][718]
      * [`AttackedText.column_labels`][719]
      * [`AttackedText.newly_swapped_words`][720]
      * [`AttackedText.num_words`][721]
      * [`AttackedText.text`][722]
      * [`AttackedText.tokenizer_input`][723]
      * [`AttackedText.words`][724]
      * [`AttackedText.words_per_input`][725]
    * [Misc Checkpoints][726]
    * [`AttackCheckpoint`][727]
      
      * [`AttackCheckpoint.load()`][728]
      * [`AttackCheckpoint.save()`][729]
      * [`AttackCheckpoint.dataset_offset`][730]
      * [`AttackCheckpoint.datetime`][731]
      * [`AttackCheckpoint.num_failed_attacks`][732]
      * [`AttackCheckpoint.num_maximized_attacks`][733]
      * [`AttackCheckpoint.num_remaining_attacks`][734]
      * [`AttackCheckpoint.num_skipped_attacks`][735]
      * [`AttackCheckpoint.num_successful_attacks`][736]
      * [`AttackCheckpoint.results_count`][737]
    * [Shared data fields][738]
    * [Misc Validators][739]
    * [`transformation_consists_of()`][740]
    * [`transformation_consists_of_word_swaps()`][741]
    * [`transformation_consists_of_word_swaps_and_deletions()`][742]
    * [`transformation_consists_of_word_swaps_differential_evolution()`][743]
    * [`validate_model_goal_function_compatibility()`][744]
    * [`validate_model_gradient_word_swap_compatibility()`][745]
    * [Shared loads word embeddings and related distances][746]
    * [`AbstractWordEmbedding`][747]
      
      * [`AbstractWordEmbedding.get_cos_sim()`][748]
      * [`AbstractWordEmbedding.get_mse_dist()`][749]
      * [`AbstractWordEmbedding.index2word()`][750]
      * [`AbstractWordEmbedding.nearest_neighbours()`][751]
      * [`AbstractWordEmbedding.word2index()`][752]
    * [`GensimWordEmbedding`][753]
      
      * [`GensimWordEmbedding.get_cos_sim()`][754]
      * [`GensimWordEmbedding.get_mse_dist()`][755]
      * [`GensimWordEmbedding.index2word()`][756]
      * [`GensimWordEmbedding.nearest_neighbours()`][757]
      * [`GensimWordEmbedding.word2index()`][758]
    * [`WordEmbedding`][759]
      
      * [`WordEmbedding.counterfitted_GLOVE_embedding()`][760]
      * [`WordEmbedding.get_cos_sim()`][761]
      * [`WordEmbedding.get_mse_dist()`][762]
      * [`WordEmbedding.index2word()`][763]
      * [`WordEmbedding.nearest_neighbours()`][764]
      * [`WordEmbedding.word2index()`][765]
      * [`WordEmbedding.PATH`][766]
* [textattack.transformations package][767]
  
  * [Transformations][768]
  * [Subpackages][769]
    
    * [textattack.transformations.sentence_transformations package][770]
      
      * [sentence_transformations package][771]
      * [Submodules][772]
        
        * [BackTranscription class][773]
        * [`BackTranscription`][774]
          
          * [`BackTranscription.back_transcribe()`][775]
        * [BackTranslation class][776]
        * [`BackTranslation`][777]
          
          * [`BackTranslation.translate()`][778]
        * [SentenceTransformation class][779]
        * [`SentenceTransformation`][780]
    * [textattack.transformations.word_insertions package][781]
      
      * [word_insertions package][782]
      * [Submodules][783]
        
        * [WordInsertion Class][784]
        * [`WordInsertion`][785]
        * [WordInsertionMaskedLM Class][786]
        * [`WordInsertionMaskedLM`][787]
          
          * [`WordInsertionMaskedLM.extra_repr_keys()`][788]
        * [WordInsertionRandomSynonym Class][789]
        * [`WordInsertionRandomSynonym`][790]
          
          * [`WordInsertionRandomSynonym.deterministic`][791]
        * [`check_if_one_word()`][792]
    * [textattack.transformations.word_merges package][793]
      
      * [word_merges package][794]
      * [Submodules][795]
        
        * [Word Merge][796]
        * [`WordMerge`][797]
        * [WordMergeMaskedLM class][798]
        * [`WordMergeMaskedLM`][799]
          
          * [`WordMergeMaskedLM.extra_repr_keys()`][800]
        * [`find_merge_index()`][801]
    * [textattack.transformations.word_swaps package][802]
      
      * [word_swaps package][803]
      * [Subpackages][804]
        
        * [textattack.transformations.word_swaps.chn_transformations package][805]
          
          * [chinese_transformations package][806]
          * [Submodules][807]
      * [Submodules][808]
        
        * [Word Swap][809]
        * [`WordSwap`][810]
        * [Word Swap by Changing Location][811]
        * [`WordSwapChangeLocation`][812]
        * [`idx_to_words()`][813]
        * [Word Swap by Changing Name][814]
        * [`WordSwapChangeName`][815]
        * [Word Swap by Changing Number][816]
        * [`WordSwapChangeNumber`][817]
        * [`idx_to_words()`][818]
        * [Word Swap by Contraction][819]
        * [`WordSwapContract`][820]
          
          * [`WordSwapContract.reverse_contraction_map`][821]
        * [Word Swap by Invisible Deletions][822]
        * [`WordSwapDeletions`][823]
          
          * [`WordSwapDeletions.apply_perturbation()`][824]
          * [`WordSwapDeletions.extra_repr_keys()`][825]
          * [`WordSwapDeletions.deterministic`][826]
        * [Word Swap for Differential Evolution][827]
        * [`WordSwapDifferentialEvolution`][828]
          
          * [`WordSwapDifferentialEvolution.apply_perturbation()`][829]
          * [`WordSwapDifferentialEvolution.get_bounds_and_precomputed()`][830]
        * [Word Swap by Embedding][831]
        * [`WordSwapEmbedding`][832]
          
          * [`WordSwapEmbedding.extra_repr_keys()`][833]
        * [`recover_word_case()`][834]
        * [Word Swap by Extension][835]
        * [`WordSwapExtend`][836]
        * [Word Swap by Gradient][837]
        * [`WordSwapGradientBased`][838]
          
          * [`WordSwapGradientBased.extra_repr_keys()`][839]
        * [Word Swap by Homoglyph][840]
        * [`WordSwapHomoglyphSwap`][841]
          
          * [`WordSwapHomoglyphSwap.apply_perturbation()`][842]
          * [`WordSwapHomoglyphSwap.extra_repr_keys()`][843]
          * [`WordSwapHomoglyphSwap.deterministic`][844]
        * [Word Swap by OpenHowNet][845]
        * [`WordSwapHowNet`][846]
          
          * [`WordSwapHowNet.extra_repr_keys()`][847]
          * [`WordSwapHowNet.PATH`][848]
        * [`recover_word_case()`][849]
        * [Word Swap by inflections][850]
        * [`WordSwapInflections`][851]
        * [Word Swap by Invisible Characters][852]
        * [`WordSwapInvisibleCharacters`][853]
          
          * [`WordSwapInvisibleCharacters.apply_perturbation()`][854]
          * [`WordSwapInvisibleCharacters.extra_repr_keys()`][855]
          * [`WordSwapInvisibleCharacters.deterministic`][856]
        * [Word Swap by BERT-Masked LM.][857]
        * [`WordSwapMaskedLM`][858]
          
          * [`WordSwapMaskedLM.extra_repr_keys()`][859]
        * [`recover_word_case()`][860]
        * [Word Swap by Neighboring Character Swap][861]
        * [`WordSwapNeighboringCharacterSwap`][862]
          
          * [`WordSwapNeighboringCharacterSwap.extra_repr_keys()`][863]
          * [`WordSwapNeighboringCharacterSwap.deterministic`][864]
        * [Word Swap by swaps characters with QWERTY adjacent keys][865]
        * [`WordSwapQWERTY`][866]
          
          * [`WordSwapQWERTY.deterministic`][867]
        * [Word Swap by Random Character Deletion][868]
        * [`WordSwapRandomCharacterDeletion`][869]
          
          * [`WordSwapRandomCharacterDeletion.extra_repr_keys()`][870]
          * [`WordSwapRandomCharacterDeletion.deterministic`][871]
        * [Word Swap by Random Character Insertion][872]
        * [`WordSwapRandomCharacterInsertion`][873]
          
          * [`WordSwapRandomCharacterInsertion.extra_repr_keys()`][874]
          * [`WordSwapRandomCharacterInsertion.deterministic`][875]
        * [Word Swap by Random Character Substitution][876]
        * [`WordSwapRandomCharacterSubstitution`][877]
          
          * [`WordSwapRandomCharacterSubstitution.extra_repr_keys()`][878]
          * [`WordSwapRandomCharacterSubstitution.deterministic`][879]
        * [Word Swap by Invisible Reorderings][880]
        * [`WordSwapReorderings`][881]
          
          * [`WordSwapReorderings.apply_perturbation()`][882]
          * [`WordSwapReorderings.extra_repr_keys()`][883]
          * [`WordSwapReorderings.deterministic`][884]
        * [Word Swap by swapping synonyms in WordNet][885]
        * [`WordSwapWordNet`][886]
  * [Submodules][887]
    
    * [Composite Transformation][888]
    * [`CompositeTransformation`][889]
    * [Transformation Abstract Class][890]
    * [`Transformation`][891]
      
      * [`Transformation.deterministic`][892]
    * [word deletion Transformation][893]
    * [`WordDeletion`][894]
    * [Word Swap Transformation by swapping the order of words][895]
    * [`WordInnerSwapRandom`][896]
      
      * [`WordInnerSwapRandom.deterministic`][897]

## Submodules[][898]

### Attack Class[][899]

* *class *textattack.attack.Attack(*goal_function: [GoalFunction][900]*, *constraints:
List[[Constraint][901] | [PreTransformationConstraint][902]]*, *transformation:
[Transformation][903]*, *search_method: [SearchMethod][904]*, *transformation_cache_size=32768*,
*constraint_cache_size=32768*)[[source]][905][][906]*
  Bases: `object`
  
  An attack generates adversarial examples on text.
  
  An attack is comprised of a goal function, constraints, transformation, and a search method. Use
  [`attack()`][907] method to attack one sample at a time.
  
  *Parameters:*
    * **goal_function** ([`GoalFunction`][908]) – A function for determining how well a perturbation
      is doing at achieving the attack’s goal.
    * **constraints** (list of [`Constraint`][909] or [`PreTransformationConstraint`][910]) – A list
      of constraints to add to the attack, defining which perturbations are valid.
    * **transformation** ([`Transformation`][911]) – The transformation applied at each step of the
      attack.
    * **search_method** ([`SearchMethod`][912]) – The method for exploring the search space of
      possible perturbations
    * **transformation_cache_size** (`int`, optional, defaults to `2**15`) – The number of items to
      keep in the transformations cache
    * **constraint_cache_size** (`int`, optional, defaults to `2**15`) – The number of items to keep
      in the constraints cache
  
  Example:
  
  >>> import textattack
  >>> import transformers
  
  >>> # Load model, tokenizer, and model_wrapper
  >>> model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-
  uncased-imdb")
  >>> tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
  >>> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
  
  >>> # Construct our four components for `Attack`
  >>> from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
  >>> from textattack.constraints.semantics import WordEmbeddingDistance
  >>> from textattack.transformations import WordSwapEmbedding
  >>> from textattack.search_methods import GreedyWordSwapWIR
  
  >>> goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
  >>> constraints = [
  ...     RepeatModification(),
  ...     StopwordModification(),
  ...     WordEmbeddingDistance(min_cos_sim=0.9)
  ... ]
  >>> transformation = WordSwapEmbedding(max_candidates=50)
  >>> search_method = GreedyWordSwapWIR(wir_method="delete")
  
  >>> # Construct the actual attack
  >>> attack = textattack.Attack(goal_function, constraints, transformation, search_method)
  
  >>> input_text = "I really enjoyed the new movie that came out last month."
  >>> label = 1 #Positive
  >>> attack_result = attack.attack(input_text, label)
  
  * attack(*example*, *ground_truth_output*)[[source]][913][][914]*
    Attack a single example.
    
    *Parameters:*
      * **example** (`str`, `OrderedDict[str, str]` or `AttackedText`) – Example to attack. It can
        be a single string or an OrderedDict where keys represent the input fields (e.g. “premise”,
        “hypothesis”) and the values are the actual input textx. Also accepts `AttackedText` that
        wraps around the input.
      * **ground_truth_output** (`int`, `float` or `str`) – Ground truth output of example. For
        classification tasks, it should be an integer representing the ground truth label. For
        regression tasks (e.g. STS), it should be the target value. For seq2seq tasks (e.g.
        translation), it should be the target string.
    *Returns:*
      [`AttackResult`][915] that represents the result of the attack.
  
  * clear_cache(*recursive=True*)[[source]][916][][917]*
  
  * cpu_()[[source]][918][][919]*
    Move any torch.nn.Module models that are part of Attack to CPU.
  
  * cuda_()[[source]][920][][921]*
    Move any torch.nn.Module models that are part of Attack to GPU.
  
  * filter_transformations(*transformed_texts*, *current_text*,
  *original_text=None*)[[source]][922][][923]*
    Filters a list of potential transformed texts based on `self.constraints` Utilizes an LRU cache
    to attempt to avoid recomputing common transformations.
    
    *Parameters:*
      * **transformed_texts** – A list of candidate transformed `AttackedText` to filter.
      * **current_text** – The current `AttackedText` on which the transformation was applied.
      * **original_text** – The original `AttackedText` from which the attack started.
  
  * get_indices_to_order(*current_text*, ***kwargs*)[[source]][924][][925]*
    Applies `pre_transformation_constraints` to `text` to get all the indices that can be used to
    search and order.
    
    *Parameters:*
      **current_text** – The current `AttackedText` for which we need to find indices are eligible
      to be ordered.
    *Returns:*
      The length and the filtered list of indices which search methods can use to search/order.
  
  * get_transformations(*current_text*, *original_text=None*, ***kwargs*)[[source]][926][][927]*
    Applies `self.transformation` to `text`, then filters the list of possible transformations
    through the applicable constraints.
    
    *Parameters:*
      * **current_text** – The current `AttackedText` on which to perform the transformations.
      * **original_text** – The original `AttackedText` from which the attack started.
    *Returns:*
      A filtered list of transformations where each transformation matches the constraints

### AttackArgs Class[][928]

* *class *textattack.attack_args.AttackArgs(*num_examples: int = 10*, *num_successful_examples: int
| None = None*, *num_examples_offset: int = 0*, *attack_n: bool = False*, *shuffle: bool = False*,
*query_budget: int | None = None*, *checkpoint_interval: int | None = None*, *checkpoint_dir: str =
'checkpoints'*, *random_seed: int = 765*, *parallel: bool = False*, *num_workers_per_device: int =
1*, *log_to_txt: str | None = None*, *log_to_csv: str | None = None*, *log_summary_to_json: str |
None = None*, *csv_coloring_style: str = 'file'*, *log_to_visdom: dict | None = None*,
*log_to_wandb: dict | None = None*, *disable_stdout: bool = False*, *silent: bool = False*,
*enable_advance_metrics: bool = False*, *metrics: Dict | None = None*)[[source]][929][][930]*
  Bases: `object`
  
  Attack arguments to be passed to [`Attacker`][931].
  
  *Parameters:*
    * **num_examples** (`int`, ‘optional`, defaults to `10`) – The number of examples to attack.
      `-1` for entire dataset.
    * **num_successful_examples** (`int`, optional, defaults to `None`) –
      
      The number of successful adversarial examples we want. This is different from
      [`num_examples`][932] as [`num_examples`][933] only cares about attacking N samples while
      [`num_successful_examples`][934] aims to keep attacking until we have N successful cases. ..
      note:
      
      If set, this argument overrides `num_examples` argument.
    * **(** (*num_examples_offset*) – obj: int, optional, defaults to `0`): The offset index to
      start at in the dataset.
    * **attack_n** (`bool`, optional, defaults to `False`) – Whether to run attack until total of N
      examples have been attacked (and not skipped).
    * **shuffle** (`bool`, optional, defaults to `False`) – If `True`, we randomly shuffle the
      dataset before attacking. However, this avoids actually shuffling the dataset internally and
      opts for shuffling the list of indices of examples we want to attack. This means
      [`shuffle`][935] can now be used with checkpoint saving.
    * **query_budget** (`int`, optional, defaults to `None`) –
      
      The maximum number of model queries allowed per example attacked. If not set, we use the query
      budget set in the [`GoalFunction`][936] object (which by default is `float("inf")`). .. note:
      
      Setting this overwrites the query budget set in :class:`~textattack.goal_functions.GoalFunctio
      n` object.
    * **checkpoint_interval** (`int`, optional, defaults to `None`) – If set, checkpoint will be
      saved after attacking every N examples. If `None` is passed, no checkpoints will be saved.
    * **checkpoint_dir** (`str`, optional, defaults to `"checkpoints"`) – The directory to save
      checkpoint files.
    * **random_seed** (`int`, optional, defaults to `765`) – Random seed for reproducibility.
    * **parallel** (`False`, optional, defaults to `False`) – If `True`, run attack using multiple
      CPUs/GPUs.
    * **num_workers_per_device** (`int`, optional, defaults to `1`) – Number of worker processes to
      run per device in parallel mode (i.e. `parallel=True`). For example, if you are using GPUs and
      `num_workers_per_device=2`, then 2 processes will be running in each GPU.
    * **log_to_txt** (`str`, optional, defaults to `None`) – If set, save attack logs as a .txt file
      to the directory specified by this argument. If the last part of the provided path ends with
      .txt extension, it is assumed to the desired path of the log file.
    * **log_to_csv** (`str`, optional, defaults to `None`) – If set, save attack logs as a CSV file
      to the directory specified by this argument. If the last part of the provided path ends with
      .csv extension, it is assumed to the desired path of the log file.
    * **csv_coloring_style** (`str`, optional, defaults to `"file"`) – Method for choosing how to
      mark perturbed parts of the text. Options are `"file"`, `"plain"`, and `"html"`. `"file"`
      wraps perturbed parts with double brackets `[[ <text> ]]` while `"plain"` does not mark the
      text in any way.
    * **log_to_visdom** (`dict`, optional, defaults to `None`) – If set, Visdom logger is used with
      the provided dictionary passed as a keyword arguments to `VisdomLogger`. Pass in empty
      dictionary to use default arguments. For custom logger, the dictionary should have the
      following three keys and their corresponding values: `"env", "port", "hostname"`.
    * **log_to_wandb** (`dict`, optional, defaults to `None`) – If set, WandB logger is used with
      the provided dictionary passed as a keyword arguments to `WeightsAndBiasesLogger`. Pass in
      empty dictionary to use default arguments. For custom logger, the dictionary should have the
      following key and its corresponding value: `"project"`.
    * **disable_stdout** (`bool`, optional, defaults to `False`) – Disable displaying individual
      attack results to stdout.
    * **silent** (`bool`, optional, defaults to `False`) – Disable all logging (except for errors).
      This is stronger than [`disable_stdout`][937].
    * **enable_advance_metrics** (`bool`, optional, defaults to `False`) – Enable calculation and
      display of optional advance post-hoc metrics like perplexity, grammar errors, etc.
  
  * *classmethod *create_loggers_from_args(*args*)[[source]][938][][939]*
    Creates AttackLogManager from an AttackArgs object.
  
  * attack_n*: bool** = False*[][940]*
  
  * checkpoint_dir*: str** = 'checkpoints'*[][941]*
  
  * checkpoint_interval*: int** = None*[][942]*
  
  * csv_coloring_style*: str** = 'file'*[][943]*
  
  * disable_stdout*: bool** = False*[][944]*
  
  * enable_advance_metrics*: bool** = False*[][945]*
  
  * log_summary_to_json*: str** = None*[][946]*
  
  * log_to_csv*: str** = None*[][947]*
  
  * log_to_txt*: str** = None*[][948]*
  
  * log_to_visdom*: dict** = None*[][949]*
  
  * log_to_wandb*: dict** = None*[][950]*
  
  * metrics*: Dict | None** = None*[][951]*
  
  * num_examples*: int** = 10*[][952]*
  
  * num_examples_offset*: int** = 0*[][953]*
  
  * num_successful_examples*: int** = None*[][954]*
  
  * num_workers_per_device*: int** = 1*[][955]*
  
  * parallel*: bool** = False*[][956]*
  
  * query_budget*: int** = None*[][957]*
  
  * random_seed*: int** = 765*[][958]*
  
  * shuffle*: bool** = False*[][959]*
  
  * silent*: bool** = False*[][960]*

* *class *textattack.attack_args.CommandLineAttackArgs(*model: str = None*, *model_from_file: str =
None*, *model_from_huggingface: str = None*, *dataset_by_model: str = None*,
*dataset_from_huggingface: str = None*, *dataset_from_file: str = None*, *dataset_split: str =
None*, *filter_by_labels: list = None*, *transformation: str = 'word-swap-embedding'*, *constraints:
list = <factory>*, *goal_function: str = 'untargeted-classification'*, *search_method: str =
'greedy-word-wir'*, *attack_recipe: str = None*, *attack_from_file: str = None*, *interactive: bool
= False*, *parallel: bool = False*, *model_batch_size: int = 32*, *model_cache_size: int = 262144*,
*constraint_cache_size: int = 262144*, *num_examples: int = 10*, *num_successful_examples: int =
None*, *num_examples_offset: int = 0*, *attack_n: bool = False*, *shuffle: bool = False*,
*query_budget: int = None*, *checkpoint_interval: int = None*, *checkpoint_dir: str =
'checkpoints'*, *random_seed: int = 765*, *num_workers_per_device: int = 1*, *log_to_txt: str =
None*, *log_to_csv: str = None*, *log_summary_to_json: str = None*, *csv_coloring_style: str =
'file'*, *log_to_visdom: dict = None*, *log_to_wandb: dict = None*, *disable_stdout: bool = False*,
*silent: bool = False*, *enable_advance_metrics: bool = False*, *metrics: Union[Dict*, *NoneType] =
None*)[[source]][961][][962]*
  Bases: [`AttackArgs`][963], `_CommandLineAttackArgs`, [`DatasetArgs`][964], [`ModelArgs`][965]

### Attacker Class[][966]

* *class *textattack.attacker.Attacker(*attack*, *dataset*,
*attack_args=None*)[[source]][967][][968]*
  Bases: `object`
  
  Class for running attacks on a dataset with specified parameters. This class uses the
  [`Attack`][969] to actually run the attacks, while also providing useful features such as parallel
  processing, saving/resuming from a checkpint, logging to files and stdout.
  
  *Parameters:*
    * **attack** ([`Attack`][970]) – [`Attack`][971] used to actually carry out the attack.
    * **dataset** ([`Dataset`][972]) – Dataset to attack.
    * **attack_args** ([`AttackArgs`][973]) – Arguments for attacking the dataset. For default
      settings, look at the AttackArgs class.
  
  Example:
  
  >>> import textattack
  >>> import transformers
  
  >>> model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-
  uncased-imdb")
  >>> tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
  >>> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
  
  >>> attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
  >>> dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")
  
  >>> # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
  >>> attack_args = textattack.AttackArgs(
  ...     num_examples=20,
  ...     log_to_csv="log.csv",
  ...     checkpoint_interval=5,
  ...     checkpoint_dir="checkpoints",
  ...     disable_stdout=True
  ... )
  
  >>> attacker = textattack.Attacker(attack, dataset, attack_args)
  >>> attacker.attack_dataset()
  
  * attack_dataset()[[source]][974][][975]*
    Attack the dataset.
    
    *Returns:*
      `list[AttackResult]` - List of [`AttackResult`][976] obtained after attacking the given
      dataset..
  
  * *static *attack_interactive(*attack*)[[source]][977][][978]*
  
  * *classmethod *from_checkpoint(*attack*, *dataset*, *checkpoint*)[[source]][979][][980]*
    Resume attacking from a saved checkpoint. Attacker and dataset must be recovered by the user
    again, while attack args are loaded from the saved checkpoint.
    
    *Parameters:*
      * **attack** ([`Attack`][981]) – Attack object for carrying out the attack.
      * **dataset** ([`Dataset`][982]) – Dataset to attack.
      * **checkpoint** (`Union[str, :class:`~textattack.shared.AttackChecpoint`]`) – Path of saved
        checkpoint or the actual saved checkpoint.
  
  * update_attack_args(***kwargs*)[[source]][983][][984]*
    To update any attack args, pass the new argument as keyword argument to this function.
    
    Examples:
    
    >>> attacker = #some instance of Attacker
    >>> # To switch to parallel mode and increase checkpoint interval from 100 to 500
    >>> attacker.update_attack_args(parallel=True, checkpoint_interval=500)

* textattack.attacker.attack_from_queue(*attack*, *attack_args*, *num_gpus*, *first_to_start*,
*lock*, *in_queue*, *out_queue*)[[source]][985][][986]*

* textattack.attacker.pytorch_multiprocessing_workaround()[[source]][987][][988]*

* textattack.attacker.set_env_variables(*gpu_id*)[[source]][989][][990]*

### AugmenterArgs Class[][991]

* *class *textattack.augment_args.AugmenterArgs(*input_csv: str*, *output_csv: str*, *input_column:
str*, *recipe: str = 'embedding'*, *pct_words_to_swap: float = 0.1*, *transformations_per_example:
int = 2*, *random_seed: int = 42*, *exclude_original: bool = False*, *overwrite: bool = False*,
*interactive: bool = False*, *fast_augment: bool = False*, *high_yield: bool = False*,
*enable_advanced_metrics: bool = False*)[[source]][992][][993]*
  Bases: `object`
  
  Arguments for performing data augmentation.
  
  *Parameters:*
    * **input_csv** (*str*) – Path of input CSV file to augment.
    * **output_csv** (*str*) – Path of CSV file to output augmented data.
  
  * enable_advanced_metrics*: bool** = False*[][994]*
  
  * exclude_original*: bool** = False*[][995]*
  
  * fast_augment*: bool** = False*[][996]*
  
  * high_yield*: bool** = False*[][997]*
  
  * input_column*: str*[][998]*
  
  * input_csv*: str*[][999]*
  
  * interactive*: bool** = False*[][1000]*
  
  * output_csv*: str*[][1001]*
  
  * overwrite*: bool** = False*[][1002]*
  
  * pct_words_to_swap*: float** = 0.1*[][1003]*
  
  * random_seed*: int** = 42*[][1004]*
  
  * recipe*: str** = 'embedding'*[][1005]*
  
  * transformations_per_example*: int** = 2*[][1006]*

### DatasetArgs Class[][1007]

* *class *textattack.dataset_args.DatasetArgs(*dataset_by_model: str | None = None*,
*dataset_from_huggingface: str | None = None*, *dataset_from_file: str | None = None*,
*dataset_split: str | None = None*, *filter_by_labels: list | None =
None*)[[source]][1008][][1009]*
  Bases: `object`
  
  Arguments for loading dataset from command line input.
  
  * dataset_by_model*: str** = None*[][1010]*
  
  * dataset_from_file*: str** = None*[][1011]*
  
  * dataset_from_huggingface*: str** = None*[][1012]*
  
  * dataset_split*: str** = None*[][1013]*
  
  * filter_by_labels*: list** = None*[][1014]*

### ModelArgs Class[][1015]

* *class *textattack.model_args.ModelArgs(*model: str | None = None*, *model_from_file: str | None =
None*, *model_from_huggingface: str | None = None*)[[source]][1016][][1017]*
  Bases: `object`
  
  Arguments for loading base/pretrained or trained models.
  
  * model*: str** = None*[][1018]*
  
  * model_from_file*: str** = None*[][1019]*
  
  * model_from_huggingface*: str** = None*[][1020]*

### Trainer Class[][1021]

* *class *textattack.trainer.Trainer(*model_wrapper*, *task_type='classification'*, *attack=None*,
*train_dataset=None*, *eval_dataset=None*, *training_args=None*)[[source]][1022][][1023]*
  Bases: `object`
  
  Trainer is training and eval loop for adversarial training.
  
  It is designed to work with PyTorch and Transformers models.
  
  *Parameters:*
    * **model_wrapper** (`ModelWrapper`) – Model wrapper containing both the model and the
      tokenizer.
    * **task_type** (`str`, optional, defaults to `"classification"`) – The task that the model is
      trained to perform. Currently, [`Trainer`][1024] supports two tasks: (1) `"classification"`,
      (2) `"regression"`.
    * **attack** ([`Attack`][1025]) – [`Attack`][1026] used to generate adversarial examples for
      training.
    * **train_dataset** ([`Dataset`][1027]) – Dataset for training.
    * **eval_dataset** ([`Dataset`][1028]) – Dataset for evaluation
    * **training_args** ([`TrainingArgs`][1029]) – Arguments for training.
  
  Example:
  
  >>> import textattack
  >>> import transformers
  
  >>> model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
  >>> tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
  >>> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
  
  >>> # We only use DeepWordBugGao2018 to demonstration purposes.
  >>> attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
  >>> train_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="train")
  >>> eval_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")
  
  >>> # Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learnin
  g rate of 5e-5, and effective batch size of 32 (8x4).
  >>> training_args = textattack.TrainingArgs(
  ...     num_epochs=3,
  ...     num_clean_epochs=1,
  ...     num_train_adv_examples=1000,
  ...     learning_rate=5e-5,
  ...     per_device_train_batch_size=8,
  ...     gradient_accumulation_steps=4,
  ...     log_to_tb=True,
  ... )
  
  >>> trainer = textattack.Trainer(
  ...     model_wrapper,
  ...     "classification",
  ...     attack,
  ...     train_dataset,
  ...     eval_dataset,
  ...     training_args
  ... )
  >>> trainer.train()
  
  Note
  
  When using [`Trainer`][1030] with parallel=True in [`TrainingArgs`][1031], make sure to protect
  the “entry point” of the program by using `if __name__ == '__main__':`. If not, each worker
  process used for generating adversarial examples will execute the training code again.
  
  * evaluate()[[source]][1032][][1033]*
    Evaluate the model on given evaluation dataset.
  
  * evaluate_step(*model*, *tokenizer*, *batch*)[[source]][1034][][1035]*
    Perform a single evaluation step on a batch of inputs.
    
    *Parameters:*
      * **model** (`torch.nn.Module`) – Model to train.
      * **tokenizer** – Tokenizer used to tokenize input text.
      * **batch** (`tuple[list[str], torch.Tensor]`) –
        
        By default, this will be a tuple of input texts and target tensors.
        
        Note
        
        If you override the [`get_eval_dataloader()`][1036] method, then shape/type of `batch` will
        depend on how you created your batch.
    *Returns:*
      `tuple[torch.Tensor, torch.Tensor]` where
      
      * **preds**: `torch.FloatTensor` of model’s prediction for the batch.
      * **targets**: `torch.Tensor` of model’s targets (e.g. labels, target values).
  
  * get_eval_dataloader(*dataset*, *batch_size*)[[source]][1037][][1038]*
    Returns the `torch.utils.data.DataLoader` for evaluation.
    
    *Parameters:*
      * **dataset** ([`Dataset`][1039]) – Dataset to use for evaluation.
      * **batch_size** (`int`) – Batch size for evaluation.
    *Returns:*
      `torch.utils.data.DataLoader`
  
  * get_optimizer_and_scheduler(*model*, *num_training_steps*)[[source]][1040][][1041]*
    Returns optimizer and scheduler to use for training. If you are overriding this method and do
    not want to use a scheduler, simply return `None` for scheduler.
    
    *Parameters:*
      * **model** (`torch.nn.Module`) – Model to be trained. Pass its parameters to optimizer for
        training.
      * **num_training_steps** (`int`) – Number of total training steps.
    *Returns:*
      Tuple of optimizer and scheduler `tuple[torch.optim.Optimizer,
      torch.optim.lr_scheduler._LRScheduler]`
  
  * get_train_dataloader(*dataset*, *adv_dataset*, *batch_size*)[[source]][1042][][1043]*
    Returns the `torch.utils.data.DataLoader` for training.
    
    *Parameters:*
      * **dataset** ([`Dataset`][1044]) – Original training dataset.
      * **adv_dataset** ([`Dataset`][1045]) – Adversarial examples generated from the original
        training dataset. `None` if no adversarial attack takes place.
      * **batch_size** (`int`) – Batch size for training.
    *Returns:*
      `torch.utils.data.DataLoader`
  
  * train()[[source]][1046][][1047]*
    Train the model on given training dataset.
  
  * training_step(*model*, *tokenizer*, *batch*)[[source]][1048][][1049]*
    Perform a single training step on a batch of inputs.
    
    *Parameters:*
      * **model** (`torch.nn.Module`) – Model to train.
      * **tokenizer** – Tokenizer used to tokenize input text.
      * **batch** (`tuple[list[str], torch.Tensor, torch.Tensor]`) –
        
        By default, this will be a tuple of input texts, targets, and boolean tensor indicating if
        the sample is an adversarial example.
        
        Note
        
        If you override the [`get_train_dataloader()`][1050] method, then shape/type of `batch` will
        depend on how you created your batch.
    *Returns:*
      `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` where
      
      * **loss**: `torch.FloatTensor` of shape 1 containing the loss.
      * **preds**: `torch.FloatTensor` of model’s prediction for the batch.
      * **targets**: `torch.Tensor` of model’s targets (e.g. labels, target values).

### TrainingArgs Class[][1051]

* *class *textattack.training_args.CommandLineTrainingArgs(*model_name_or_path: str*, *attack: str*,
*dataset: str*, *task_type: str = 'classification'*, *model_max_length: int = None*,
*model_num_labels: int = None*, *dataset_train_split: str = None*, *dataset_eval_split: str = None*,
*filter_train_by_labels: list = None*, *filter_eval_by_labels: list = None*, *num_epochs: int = 3*,
*num_clean_epochs: int = 1*, *attack_epoch_interval: int = 1*, *early_stopping_epochs: int = None*,
*learning_rate: float = 5e-05*, *num_warmup_steps: Union[int*, *float] = 500*, *weight_decay: float
= 0.01*, *per_device_train_batch_size: int = 8*, *per_device_eval_batch_size: int = 32*,
*gradient_accumulation_steps: int = 1*, *random_seed: int = 786*, *parallel: bool = False*,
*load_best_model_at_end: bool = False*, *alpha: float = 1.0*, *num_train_adv_examples: Union[int*,
*float] = -1*, *query_budget_train: int = None*, *attack_num_workers_per_device: int = 1*,
*output_dir: str = <factory>*, *checkpoint_interval_steps: int = None*, *checkpoint_interval_epochs:
int = None*, *save_last: bool = True*, *log_to_tb: bool = False*, *tb_log_dir: str = None*,
*log_to_wandb: bool = False*, *wandb_project: str = 'textattack'*, *logging_interval_step: int =
1*)[[source]][1052][][1053]*
  Bases: [`TrainingArgs`][1054], `_CommandLineTrainingArgs`
  
  * output_dir*: str*[][1055]*

* *class *textattack.training_args.TrainingArgs(*num_epochs: int = 3*, *num_clean_epochs: int = 1*,
*attack_epoch_interval: int = 1*, *early_stopping_epochs: int | None = None*, *learning_rate: float
= 5e-05*, *num_warmup_steps: int | float = 500*, *weight_decay: float = 0.01*,
*per_device_train_batch_size: int = 8*, *per_device_eval_batch_size: int = 32*,
*gradient_accumulation_steps: int = 1*, *random_seed: int = 786*, *parallel: bool = False*,
*load_best_model_at_end: bool = False*, *alpha: float = 1.0*, *num_train_adv_examples: int | float =
-1*, *query_budget_train: int | None = None*, *attack_num_workers_per_device: int = 1*, *output_dir:
str = <factory>*, *checkpoint_interval_steps: int | None = None*, *checkpoint_interval_epochs: int |
None = None*, *save_last: bool = True*, *log_to_tb: bool = False*, *tb_log_dir: str | None = None*,
*log_to_wandb: bool = False*, *wandb_project: str = 'textattack'*, *logging_interval_step: int =
1*)[[source]][1056][][1057]*
  Bases: `object`
  
  Arguments for `Trainer` class that is used for adversarial training.
  
  *Parameters:*
    * **num_epochs** (`int`, optional, defaults to `3`) – Total number of epochs for training.
    * **num_clean_epochs** (`int`, optional, defaults to `1`) – Number of epochs to train on just
      the original training dataset before adversarial training.
    * **attack_epoch_interval** (`int`, optional, defaults to `1`) – Generate a new adversarial
      training set every N epochs.
    * **early_stopping_epochs** (`int`, optional, defaults to `None`) – Number of epochs validation
      must increase before stopping early (`None` for no early stopping).
    * **learning_rate** (`float`, optional, defaults to `5e-5`) – Learning rate for optimizer.
    * **num_warmup_steps** (`int` or `float`, optional, defaults to `500`) – The number of steps for
      the warmup phase of linear scheduler. If [`num_warmup_steps`][1058] is a `float` between 0 and
      1, the number of warmup steps will be `math.ceil(num_training_steps * num_warmup_steps)`.
    * **weight_decay** (`float`, optional, defaults to `0.01`) – Weight decay (L2 penalty).
    * **per_device_train_batch_size** (`int`, optional, defaults to `8`) – The batch size per
      GPU/CPU for training.
    * **per_device_eval_batch_size** (`int`, optional, defaults to `32`) – The batch size per
      GPU/CPU for evaluation.
    * **gradient_accumulation_steps** (`int`, optional, defaults to `1`) – Number of updates steps
      to accumulate the gradients before performing a backward/update pass.
    * **random_seed** (`int`, optional, defaults to `786`) – Random seed for reproducibility.
    * **parallel** (`bool`, optional, defaults to `False`) – If `True`, train using multiple GPUs
      using `torch.DataParallel`.
    * **load_best_model_at_end** (`bool`, optional, defaults to `False`) – If `True`, keep track of
      the best model across training and load it at the end.
    * **alpha** (`float`, optional, defaults to `1.0`) – The weight for adversarial loss.
    * **num_train_adv_examples** (`int` or `float`, optional, defaults to `-1`) – The number of
      samples to successfully attack when generating adversarial training set before start of every
      epoch. If [`num_train_adv_examples`][1059] is a `float` between 0 and 1, the number of
      adversarial examples generated is fraction of the original training set.
    * **query_budget_train** (`int`, optional, defaults to `None`) – The max query budget to use
      when generating adversarial training set. `None` means infinite query budget.
    * **attack_num_workers_per_device** (`int`, defaults to optional, `1`) – Number of worker
      processes to run per device for attack. Same as `num_workers_per_device` argument for
      [`AttackArgs`][1060].
    * **output_dir** (`str`, optional) – Directory to output training logs and checkpoints. Defaults
      to `/outputs/%Y-%m-%d-%H-%M-%S-%f` format.
    * **checkpoint_interval_steps** (`int`, optional, defaults to `None`) – If set, save model
      checkpoint after every N updates to the model.
    * **checkpoint_interval_epochs** (`int`, optional, defaults to `None`) – If set, save model
      checkpoint after every N epochs.
    * **save_last** (`bool`, optional, defaults to `True`) – If `True`, save the model at end of
      training. Can be used with [`load_best_model_at_end`][1061] to save the best model at the end.
    * **log_to_tb** (`bool`, optional, defaults to `False`) – If `True`, log to Tensorboard.
    * **tb_log_dir** (`str`, optional, defaults to `"./runs"`) – Path of Tensorboard log directory.
    * **log_to_wandb** (`bool`, optional, defaults to `False`) – If `True`, log to Wandb.
    * **wandb_project** (`str`, optional, defaults to `"textattack"`) – Name of Wandb project for
      logging.
    * **logging_interval_step** (`int`, optional, defaults to `1`) – Log to Tensorboard/Wandb every
      N training steps.
  
  * alpha*: float** = 1.0*[][1062]*
  
  * attack_epoch_interval*: int** = 1*[][1063]*
  
  * attack_num_workers_per_device*: int** = 1*[][1064]*
  
  * checkpoint_interval_epochs*: int** = None*[][1065]*
  
  * checkpoint_interval_steps*: int** = None*[][1066]*
  
  * early_stopping_epochs*: int** = None*[][1067]*
  
  * gradient_accumulation_steps*: int** = 1*[][1068]*
  
  * learning_rate*: float** = 5e-05*[][1069]*
  
  * load_best_model_at_end*: bool** = False*[][1070]*
  
  * log_to_tb*: bool** = False*[][1071]*
  
  * log_to_wandb*: bool** = False*[][1072]*
  
  * logging_interval_step*: int** = 1*[][1073]*
  
  * num_clean_epochs*: int** = 1*[][1074]*
  
  * num_epochs*: int** = 3*[][1075]*
  
  * num_train_adv_examples*: int | float** = -1*[][1076]*
  
  * num_warmup_steps*: int | float** = 500*[][1077]*
  
  * output_dir*: str*[][1078]*
  
  * parallel*: bool** = False*[][1079]*
  
  * per_device_eval_batch_size*: int** = 32*[][1080]*
  
  * per_device_train_batch_size*: int** = 8*[][1081]*
  
  * query_budget_train*: int** = None*[][1082]*
  
  * random_seed*: int** = 786*[][1083]*
  
  * save_last*: bool** = True*[][1084]*
  
  * tb_log_dir*: str** = None*[][1085]*
  
  * wandb_project*: str** = 'textattack'*[][1086]*
  
  * weight_decay*: float** = 0.01*[][1087]*

* textattack.training_args.default_output_dir()[[source]][1088][][1089]*
[ Previous][1090] [Next ][1091]

© Copyright 2021-24, UVA QData Lab.

Built with [Sphinx][1092] using a [theme][1093] provided by [Read the Docs][1094].

[1]: ../_sources/apidoc/textattack.rst.txt
[2]: #module-textattack
[3]: https://github.com/QData/TextAttack
[4]: #subpackages
[5]: textattack.attack_recipes.html
[6]: textattack.attack_recipes.html#attack-recipes-package
[7]: textattack.attack_recipes.html#module-textattack.attack_recipes.a2t_yoo_2021
[8]: textattack.attack_recipes.html#a2t-a2t-attack-for-adversarial-training-recipe
[9]: textattack.attack_recipes.html#textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021
[10]: textattack.attack_recipes.html#textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021.build
[11]: textattack.attack_recipes.html#attack-recipe-class
[12]: textattack.attack_recipes.html#textattack.attack_recipes.attack_recipe.AttackRecipe
[13]: textattack.attack_recipes.html#textattack.attack_recipes.attack_recipe.AttackRecipe.build
[14]: textattack.attack_recipes.html#imperceptible-perturbations-algorithm
[15]: textattack.attack_recipes.html#textattack.attack_recipes.bad_characters_2021.BadCharacters2021
[16]: textattack.attack_recipes.html#textattack.attack_recipes.bad_characters_2021.BadCharacters2021
.build
[17]: textattack.attack_recipes.html#bae-bae-bert-based-adversarial-examples
[18]: textattack.attack_recipes.html#textattack.attack_recipes.bae_garg_2019.BAEGarg2019
[19]: textattack.attack_recipes.html#textattack.attack_recipes.bae_garg_2019.BAEGarg2019.build
[20]: textattack.attack_recipes.html#bert-attack
[21]: textattack.attack_recipes.html#textattack.attack_recipes.bert_attack_li_2020.BERTAttackLi2020
[22]: textattack.attack_recipes.html#textattack.attack_recipes.bert_attack_li_2020.BERTAttackLi2020.
build
[23]: textattack.attack_recipes.html#checklist
[24]: textattack.attack_recipes.html#textattack.attack_recipes.checklist_ribeiro_2020.CheckList2020
[25]: textattack.attack_recipes.html#textattack.attack_recipes.checklist_ribeiro_2020.CheckList2020.
build
[26]: textattack.attack_recipes.html#attack-chinese-recipe
[27]: textattack.attack_recipes.html#textattack.attack_recipes.chinese_recipe.ChineseRecipe
[28]: textattack.attack_recipes.html#textattack.attack_recipes.chinese_recipe.ChineseRecipe.build
[29]: textattack.attack_recipes.html#clare-recipe
[30]: textattack.attack_recipes.html#textattack.attack_recipes.clare_li_2020.CLARE2020
[31]: textattack.attack_recipes.html#textattack.attack_recipes.clare_li_2020.CLARE2020.build
[32]: textattack.attack_recipes.html#deepwordbug
[33]: textattack.attack_recipes.html#textattack.attack_recipes.deepwordbug_gao_2018.DeepWordBugGao20
18
[34]: textattack.attack_recipes.html#textattack.attack_recipes.deepwordbug_gao_2018.DeepWordBugGao20
18.build
[35]: textattack.attack_recipes.html#faster-alzantot-genetic-algorithm
[36]: textattack.attack_recipes.html#textattack.attack_recipes.faster_genetic_algorithm_jia_2019.Fas
terGeneticAlgorithmJia2019
[37]: textattack.attack_recipes.html#textattack.attack_recipes.faster_genetic_algorithm_jia_2019.Fas
terGeneticAlgorithmJia2019.build
[38]: textattack.attack_recipes.html#attack-french-recipe
[39]: textattack.attack_recipes.html#textattack.attack_recipes.french_recipe.FrenchRecipe
[40]: textattack.attack_recipes.html#textattack.attack_recipes.french_recipe.FrenchRecipe.build
[41]: textattack.attack_recipes.html#alzantot-genetic-algorithm
[42]: textattack.attack_recipes.html#textattack.attack_recipes.genetic_algorithm_alzantot_2018.Genet
icAlgorithmAlzantot2018
[43]: textattack.attack_recipes.html#textattack.attack_recipes.genetic_algorithm_alzantot_2018.Genet
icAlgorithmAlzantot2018.build
[44]: textattack.attack_recipes.html#hotflip
[45]: textattack.attack_recipes.html#textattack.attack_recipes.hotflip_ebrahimi_2017.HotFlipEbrahimi
2017
[46]: textattack.attack_recipes.html#textattack.attack_recipes.hotflip_ebrahimi_2017.HotFlipEbrahimi
2017.build
[47]: textattack.attack_recipes.html#improved-genetic-algorithm
[48]: textattack.attack_recipes.html#textattack.attack_recipes.iga_wang_2019.IGAWang2019
[49]: textattack.attack_recipes.html#textattack.attack_recipes.iga_wang_2019.IGAWang2019.build
[50]: textattack.attack_recipes.html#input-reduction
[51]: textattack.attack_recipes.html#textattack.attack_recipes.input_reduction_feng_2018.InputReduct
ionFeng2018
[52]: textattack.attack_recipes.html#textattack.attack_recipes.input_reduction_feng_2018.InputReduct
ionFeng2018.build
[53]: textattack.attack_recipes.html#kuleshov2017
[54]: textattack.attack_recipes.html#textattack.attack_recipes.kuleshov_2017.Kuleshov2017
[55]: textattack.attack_recipes.html#textattack.attack_recipes.kuleshov_2017.Kuleshov2017.build
[56]: textattack.attack_recipes.html#morpheus2020
[57]: textattack.attack_recipes.html#textattack.attack_recipes.morpheus_tan_2020.MorpheusTan2020
[58]: textattack.attack_recipes.html#textattack.attack_recipes.morpheus_tan_2020.MorpheusTan2020.bui
ld
[59]: textattack.attack_recipes.html#pruthi2019-combating-with-robust-word-recognition
[60]: textattack.attack_recipes.html#textattack.attack_recipes.pruthi_2019.Pruthi2019
[61]: textattack.attack_recipes.html#textattack.attack_recipes.pruthi_2019.Pruthi2019.build
[62]: textattack.attack_recipes.html#particle-swarm-optimization
[63]: textattack.attack_recipes.html#textattack.attack_recipes.pso_zang_2020.PSOZang2020
[64]: textattack.attack_recipes.html#textattack.attack_recipes.pso_zang_2020.PSOZang2020.build
[65]: textattack.attack_recipes.html#pwws
[66]: textattack.attack_recipes.html#textattack.attack_recipes.pwws_ren_2019.PWWSRen2019
[67]: textattack.attack_recipes.html#textattack.attack_recipes.pwws_ren_2019.PWWSRen2019.build
[68]: textattack.attack_recipes.html#seq2sick
[69]: textattack.attack_recipes.html#textattack.attack_recipes.seq2sick_cheng_2018_blackbox.Seq2Sick
Cheng2018BlackBox
[70]: textattack.attack_recipes.html#textattack.attack_recipes.seq2sick_cheng_2018_blackbox.Seq2Sick
Cheng2018BlackBox.build
[71]: textattack.attack_recipes.html#attack-spanish-recipe
[72]: textattack.attack_recipes.html#textattack.attack_recipes.spanish_recipe.SpanishRecipe
[73]: textattack.attack_recipes.html#textattack.attack_recipes.spanish_recipe.SpanishRecipe.build
[74]: textattack.attack_recipes.html#textbugger
[75]: textattack.attack_recipes.html#textattack.attack_recipes.textbugger_li_2018.TextBuggerLi2018
[76]: textattack.attack_recipes.html#textattack.attack_recipes.textbugger_li_2018.TextBuggerLi2018.b
uild
[77]: textattack.attack_recipes.html#textfooler-is-bert-really-robust
[78]: textattack.attack_recipes.html#textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019
[79]: textattack.attack_recipes.html#textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019
.build
[80]: textattack.attack_results.html
[81]: textattack.attack_results.html#id1
[82]: textattack.attack_results.html#module-textattack.attack_results.attack_result
[83]: textattack.attack_results.html#attackresult-class
[84]: textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult
[85]: textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.diff_color
[86]: textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.goal_funct
ion_result_str
[87]: textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.original_t
ext
[88]: textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.perturbed_
text
[89]: textattack.attack_results.html#textattack.attack_results.attack_result.AttackResult.str_lines
[90]: textattack.attack_results.html#failedattackresult-class
[91]: textattack.attack_results.html#textattack.attack_results.failed_attack_result.FailedAttackResu
lt
[92]: textattack.attack_results.html#textattack.attack_results.failed_attack_result.FailedAttackResu
lt.goal_function_result_str
[93]: textattack.attack_results.html#textattack.attack_results.failed_attack_result.FailedAttackResu
lt.str_lines
[94]: textattack.attack_results.html#maximizedattackresult-class
[95]: textattack.attack_results.html#textattack.attack_results.maximized_attack_result.MaximizedAtta
ckResult
[96]: textattack.attack_results.html#skippedattackresult-class
[97]: textattack.attack_results.html#textattack.attack_results.skipped_attack_result.SkippedAttackRe
sult
[98]: textattack.attack_results.html#textattack.attack_results.skipped_attack_result.SkippedAttackRe
sult.goal_function_result_str
[99]: textattack.attack_results.html#textattack.attack_results.skipped_attack_result.SkippedAttackRe
sult.str_lines
[100]: textattack.attack_results.html#successfulattackresult-class
[101]: textattack.attack_results.html#textattack.attack_results.successful_attack_result.SuccessfulA
ttackResult
[102]: textattack.augmentation.html
[103]: textattack.augmentation.html#augmentation
[104]: textattack.augmentation.html#module-textattack.augmentation.augmenter
[105]: textattack.augmentation.html#augmenter-class
[106]: textattack.augmentation.html#textattack.augmentation.augmenter.AugmentationResult
[107]: textattack.augmentation.html#textattack.augmentation.augmenter.AugmentationResult.tempResult
[108]: textattack.augmentation.html#textattack.augmentation.augmenter.Augmenter
[109]: textattack.augmentation.html#textattack.augmentation.augmenter.Augmenter.augment
[110]: textattack.augmentation.html#textattack.augmentation.augmenter.Augmenter.augment_many
[111]: textattack.augmentation.html#textattack.augmentation.augmenter.Augmenter.augment_text_with_id
s
[112]: textattack.augmentation.html#augmenter-recipes
[113]: textattack.augmentation.html#textattack.augmentation.recipes.BackTranscriptionAugmenter
[114]: textattack.augmentation.html#textattack.augmentation.recipes.BackTranslationAugmenter
[115]: textattack.augmentation.html#textattack.augmentation.recipes.CLAREAugmenter
[116]: textattack.augmentation.html#textattack.augmentation.recipes.CharSwapAugmenter
[117]: textattack.augmentation.html#textattack.augmentation.recipes.CheckListAugmenter
[118]: textattack.augmentation.html#textattack.augmentation.recipes.DeletionAugmenter
[119]: textattack.augmentation.html#textattack.augmentation.recipes.EasyDataAugmenter
[120]: textattack.augmentation.html#textattack.augmentation.recipes.EasyDataAugmenter.augment
[121]: textattack.augmentation.html#textattack.augmentation.recipes.EmbeddingAugmenter
[122]: textattack.augmentation.html#textattack.augmentation.recipes.SwapAugmenter
[123]: textattack.augmentation.html#textattack.augmentation.recipes.SynonymInsertionAugmenter
[124]: textattack.augmentation.html#textattack.augmentation.recipes.WordNetAugmenter
[125]: textattack.commands.html
[126]: textattack.commands.html#id1
[127]: textattack.commands.html#module-textattack.commands.attack_command
[128]: textattack.commands.html#attackcommand-class
[129]: textattack.commands.html#textattack.commands.attack_command.AttackCommand
[130]: textattack.commands.html#textattack.commands.attack_command.AttackCommand.register_subcommand
[131]: textattack.commands.html#textattack.commands.attack_command.AttackCommand.run
[132]: textattack.commands.html#attackresumecommand-class
[133]: textattack.commands.html#textattack.commands.attack_resume_command.AttackResumeCommand
[134]: textattack.commands.html#textattack.commands.attack_resume_command.AttackResumeCommand.regist
er_subcommand
[135]: textattack.commands.html#textattack.commands.attack_resume_command.AttackResumeCommand.run
[136]: textattack.commands.html#augmentcommand-class
[137]: textattack.commands.html#textattack.commands.augment_command.AugmentCommand
[138]: textattack.commands.html#textattack.commands.augment_command.AugmentCommand.register_subcomma
nd
[139]: textattack.commands.html#textattack.commands.augment_command.AugmentCommand.run
[140]: textattack.commands.html#benchmarkrecipecommand-class
[141]: textattack.commands.html#textattack.commands.benchmark_recipe_command.BenchmarkRecipeCommand
[142]: textattack.commands.html#textattack.commands.benchmark_recipe_command.BenchmarkRecipeCommand.
register_subcommand
[143]: textattack.commands.html#textattack.commands.benchmark_recipe_command.BenchmarkRecipeCommand.
run
[144]: textattack.commands.html#evalmodelcommand-class
[145]: textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand
[146]: textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand.get_preds
[147]: textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand.register_sub
command
[148]: textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand.run
[149]: textattack.commands.html#textattack.commands.eval_model_command.EvalModelCommand.test_model_o
n_dataset
[150]: textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs
[151]: textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs.batch_size
[152]: textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs.num_examples
[153]: textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs.num_examples_of
fset
[154]: textattack.commands.html#textattack.commands.eval_model_command.ModelEvalArgs.random_seed
[155]: textattack.commands.html#listthingscommand-class
[156]: textattack.commands.html#textattack.commands.list_things_command.ListThingsCommand
[157]: textattack.commands.html#textattack.commands.list_things_command.ListThingsCommand.register_s
ubcommand
[158]: textattack.commands.html#textattack.commands.list_things_command.ListThingsCommand.run
[159]: textattack.commands.html#textattack.commands.list_things_command.ListThingsCommand.things
[160]: textattack.commands.html#peekdatasetcommand-class
[161]: textattack.commands.html#textattack.commands.peek_dataset_command.PeekDatasetCommand
[162]: textattack.commands.html#textattack.commands.peek_dataset_command.PeekDatasetCommand.register
_subcommand
[163]: textattack.commands.html#textattack.commands.peek_dataset_command.PeekDatasetCommand.run
[164]: textattack.commands.html#textattack-cli-main-class
[165]: textattack.commands.html#textattack.commands.textattack_cli.main
[166]: textattack.commands.html#textattack.commands.textattack_command.TextAttackCommand
[167]: textattack.commands.html#textattack.commands.textattack_command.TextAttackCommand.register_su
bcommand
[168]: textattack.commands.html#textattack.commands.textattack_command.TextAttackCommand.run
[169]: textattack.commands.html#trainmodelcommand-class
[170]: textattack.commands.html#textattack.commands.train_model_command.TrainModelCommand
[171]: textattack.commands.html#textattack.commands.train_model_command.TrainModelCommand.register_s
ubcommand
[172]: textattack.commands.html#textattack.commands.train_model_command.TrainModelCommand.run
[173]: textattack.constraints.html
[174]: textattack.constraints.html#constraints
[175]: textattack.constraints.html#subpackages
[176]: textattack.constraints.grammaticality.html
[177]: textattack.constraints.grammaticality.html#grammaticality
[178]: textattack.constraints.grammaticality.html#subpackages
[179]: textattack.constraints.grammaticality.language_models.html
[180]: textattack.constraints.grammaticality.language_models.html#non-pre-language-models
[181]: textattack.constraints.grammaticality.language_models.html#subpackages
[182]: textattack.constraints.grammaticality.language_models.html#module-textattack.constraints.gram
maticality.language_models.gpt2
[183]: textattack.constraints.grammaticality.html#module-textattack.constraints.grammaticality.cola
[184]: textattack.constraints.grammaticality.html#cola-for-grammaticality
[185]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.cola.COLA
[186]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.cola.COLA.cl
ear_cache
[187]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.cola.COLA.ex
tra_repr_keys
[188]: textattack.constraints.grammaticality.html#languagetool-grammar-checker
[189]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.language_too
l.LanguageTool
[190]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.language_too
l.LanguageTool.extra_repr_keys
[191]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.language_too
l.LanguageTool.get_errors
[192]: textattack.constraints.grammaticality.html#part-of-speech-constraint
[193]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.part_of_spee
ch.PartOfSpeech
[194]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.part_of_spee
ch.PartOfSpeech.check_compatibility
[195]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.part_of_spee
ch.PartOfSpeech.clear_cache
[196]: textattack.constraints.grammaticality.html#textattack.constraints.grammaticality.part_of_spee
ch.PartOfSpeech.extra_repr_keys
[197]: textattack.constraints.overlap.html
[198]: textattack.constraints.overlap.html#overlap-constraints
[199]: textattack.constraints.overlap.html#module-textattack.constraints.overlap.bleu_score
[200]: textattack.constraints.overlap.html#bleu-constraints
[201]: textattack.constraints.overlap.html#textattack.constraints.overlap.bleu_score.BLEU
[202]: textattack.constraints.overlap.html#textattack.constraints.overlap.bleu_score.BLEU.extra_repr
_keys
[203]: textattack.constraints.overlap.html#chrf-constraints
[204]: textattack.constraints.overlap.html#textattack.constraints.overlap.chrf_score.chrF
[205]: textattack.constraints.overlap.html#textattack.constraints.overlap.chrf_score.chrF.extra_repr
_keys
[206]: textattack.constraints.overlap.html#edit-distance-constraints
[207]: textattack.constraints.overlap.html#textattack.constraints.overlap.levenshtein_edit_distance.
LevenshteinEditDistance
[208]: textattack.constraints.overlap.html#textattack.constraints.overlap.levenshtein_edit_distance.
LevenshteinEditDistance.extra_repr_keys
[209]: textattack.constraints.overlap.html#max-perturb-words-constraints
[210]: textattack.constraints.overlap.html#textattack.constraints.overlap.max_words_perturbed.MaxWor
dsPerturbed
[211]: textattack.constraints.overlap.html#textattack.constraints.overlap.max_words_perturbed.MaxWor
dsPerturbed.extra_repr_keys
[212]: textattack.constraints.overlap.html#meteor-constraints
[213]: textattack.constraints.overlap.html#textattack.constraints.overlap.meteor_score.METEOR
[214]: textattack.constraints.overlap.html#textattack.constraints.overlap.meteor_score.METEOR.extra_
repr_keys
[215]: textattack.constraints.pre_transformation.html
[216]: textattack.constraints.pre_transformation.html#pre-transformation
[217]: textattack.constraints.pre_transformation.html#module-textattack.constraints.pre_transformati
on.input_column_modification
[218]: textattack.constraints.pre_transformation.html#input-column-modification
[219]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.inpu
t_column_modification.InputColumnModification
[220]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.inpu
t_column_modification.InputColumnModification.extra_repr_keys
[221]: textattack.constraints.pre_transformation.html#max-modification-rate
[222]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.max_
modification_rate.MaxModificationRate
[223]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.max_
modification_rate.MaxModificationRate.extra_repr_keys
[224]: textattack.constraints.pre_transformation.html#id2
[225]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.max_
num_words_modified.MaxNumWordsModified
[226]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.max_
num_words_modified.MaxNumWordsModified.extra_repr_keys
[227]: textattack.constraints.pre_transformation.html#max-word-index-modification
[228]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.max_
word_index_modification.MaxWordIndexModification
[229]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.max_
word_index_modification.MaxWordIndexModification.extra_repr_keys
[230]: textattack.constraints.pre_transformation.html#min-word-lenth
[231]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.min_
word_length.MinWordLength
[232]: textattack.constraints.pre_transformation.html#repeat-modification
[233]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.repe
at_modification.RepeatModification
[234]: textattack.constraints.pre_transformation.html#stopword-modification
[235]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.stop
word_modification.StopwordModification
[236]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.stop
word_modification.StopwordModification.check_compatibility
[237]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.unmo
difiable_indices.UnmodifiableIndices
[238]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.unmo
difiable_indices.UnmodifiableIndices.extra_repr_keys
[239]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.unmo
difiable_phrases.UnmodifablePhrases
[240]: textattack.constraints.pre_transformation.html#textattack.constraints.pre_transformation.unmo
difiable_phrases.UnmodifablePhrases.extra_repr_keys
[241]: textattack.constraints.semantics.html
[242]: textattack.constraints.semantics.html#semantic-constraints
[243]: textattack.constraints.semantics.html#subpackages
[244]: textattack.constraints.semantics.sentence_encoders.html
[245]: textattack.constraints.semantics.sentence_encoders.html#sentence-encoder-constraint
[246]: textattack.constraints.semantics.sentence_encoders.html#subpackages
[247]: textattack.constraints.semantics.sentence_encoders.html#module-textattack.constraints.semanti
cs.sentence_encoders.sentence_encoder
[248]: textattack.constraints.semantics.html#module-textattack.constraints.semantics.bert_score
[249]: textattack.constraints.semantics.html#bert-score
[250]: textattack.constraints.semantics.html#textattack.constraints.semantics.bert_score.BERTScore
[251]: textattack.constraints.semantics.html#textattack.constraints.semantics.bert_score.BERTScore.e
xtra_repr_keys
[252]: textattack.constraints.semantics.html#textattack.constraints.semantics.bert_score.BERTScore.S
CORE_TYPE2IDX
[253]: textattack.constraints.semantics.html#word-embedding-distance
[254]: textattack.constraints.semantics.html#textattack.constraints.semantics.word_embedding_distanc
e.WordEmbeddingDistance
[255]: textattack.constraints.semantics.html#textattack.constraints.semantics.word_embedding_distanc
e.WordEmbeddingDistance.check_compatibility
[256]: textattack.constraints.semantics.html#textattack.constraints.semantics.word_embedding_distanc
e.WordEmbeddingDistance.extra_repr_keys
[257]: textattack.constraints.semantics.html#textattack.constraints.semantics.word_embedding_distanc
e.WordEmbeddingDistance.get_cos_sim
[258]: textattack.constraints.semantics.html#textattack.constraints.semantics.word_embedding_distanc
e.WordEmbeddingDistance.get_mse_dist
[259]: textattack.constraints.html#module-textattack.constraints.constraint
[260]: textattack.constraints.html#textattack-constraint-class
[261]: textattack.constraints.html#textattack.constraints.constraint.Constraint
[262]: textattack.constraints.html#textattack.constraints.constraint.Constraint.call_many
[263]: textattack.constraints.html#textattack.constraints.constraint.Constraint.check_compatibility
[264]: textattack.constraints.html#textattack.constraints.constraint.Constraint.extra_repr_keys
[265]: textattack.constraints.html#pre-transformation-constraint-class
[266]: textattack.constraints.html#textattack.constraints.pre_transformation_constraint.PreTransform
ationConstraint
[267]: textattack.constraints.html#textattack.constraints.pre_transformation_constraint.PreTransform
ationConstraint.check_compatibility
[268]: textattack.constraints.html#textattack.constraints.pre_transformation_constraint.PreTransform
ationConstraint.extra_repr_keys
[269]: textattack.datasets.html
[270]: textattack.datasets.html#datasets-package
[271]: textattack.datasets.html#subpackages
[272]: textattack.datasets.helpers.html
[273]: textattack.datasets.helpers.html#dataset-helpers
[274]: textattack.datasets.helpers.html#module-textattack.datasets.helpers.ted_multi
[275]: textattack.datasets.helpers.html#ted-multi-translationdataset-class
[276]: textattack.datasets.helpers.html#textattack.datasets.helpers.ted_multi.TedMultiTranslationDat
aset
[277]: textattack.datasets.html#module-textattack.datasets.dataset
[278]: textattack.datasets.html#dataset-class
[279]: textattack.datasets.html#textattack.datasets.dataset.Dataset
[280]: textattack.datasets.html#textattack.datasets.dataset.Dataset.filter_by_labels_
[281]: textattack.datasets.html#textattack.datasets.dataset.Dataset.shuffle
[282]: textattack.datasets.html#huggingfacedataset-class
[283]: textattack.datasets.html#textattack.datasets.huggingface_dataset.HuggingFaceDataset
[284]: textattack.datasets.html#textattack.datasets.huggingface_dataset.HuggingFaceDataset.filter_by
_labels_
[285]: textattack.datasets.html#textattack.datasets.huggingface_dataset.HuggingFaceDataset.shuffle
[286]: textattack.datasets.html#textattack.datasets.huggingface_dataset.get_datasets_dataset_columns
[287]: textattack.goal_function_results.html
[288]: textattack.goal_function_results.html#goal-function-result-package
[289]: textattack.goal_function_results.html#subpackages
[290]: textattack.goal_function_results.custom.html
[291]: textattack.goal_function_results.custom.html#custom-goal-function-result-package
[292]: textattack.goal_function_results.custom.html#module-textattack.goal_function_results.custom.l
ogit_sum_goal_function_result
[293]: textattack.goal_function_results.custom.html#logitsumgoalfunctionresult-class
[294]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.logit_su
m_goal_function_result.LogitSumGoalFunctionResult
[295]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.logit_su
m_goal_function_result.LogitSumGoalFunctionResult.get_colored_output
[296]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.logit_su
m_goal_function_result.LogitSumGoalFunctionResult.get_text_color_input
[297]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.logit_su
m_goal_function_result.LogitSumGoalFunctionResult.get_text_color_perturbed
[298]: textattack.goal_function_results.custom.html#namedentityrecognitionoalfunctionresult-class
[299]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.named_en
tity_recognition_goal_function_result.NamedEntityRecognitionGoalFunctionResult
[300]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.named_en
tity_recognition_goal_function_result.NamedEntityRecognitionGoalFunctionResult.get_colored_output
[301]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.named_en
tity_recognition_goal_function_result.NamedEntityRecognitionGoalFunctionResult.get_text_color_input
[302]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.named_en
tity_recognition_goal_function_result.NamedEntityRecognitionGoalFunctionResult.get_text_color_pertur
bed
[303]: textattack.goal_function_results.custom.html#targetedbonusgoalfunctionresult-class
[304]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.targeted
_bonus_goal_function_result.TargetedBonusGoalFunctionResult
[305]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.targeted
_bonus_goal_function_result.TargetedBonusGoalFunctionResult.get_colored_output
[306]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.targeted
_bonus_goal_function_result.TargetedBonusGoalFunctionResult.get_text_color_input
[307]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.targeted
_bonus_goal_function_result.TargetedBonusGoalFunctionResult.get_text_color_perturbed
[308]: textattack.goal_function_results.custom.html#targetedstrictgoalfunctionresult-class
[309]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.targeted
_strict_goal_function_result.TargetedStrictGoalFunctionResult
[310]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.targeted
_strict_goal_function_result.TargetedStrictGoalFunctionResult.get_colored_output
[311]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.targeted
_strict_goal_function_result.TargetedStrictGoalFunctionResult.get_text_color_input
[312]: textattack.goal_function_results.custom.html#textattack.goal_function_results.custom.targeted
_strict_goal_function_result.TargetedStrictGoalFunctionResult.get_text_color_perturbed
[313]: textattack.goal_function_results.html#module-textattack.goal_function_results.classification_
goal_function_result
[314]: textattack.goal_function_results.html#classificationgoalfunctionresult-class
[315]: textattack.goal_function_results.html#textattack.goal_function_results.classification_goal_fu
nction_result.ClassificationGoalFunctionResult
[316]: textattack.goal_function_results.html#textattack.goal_function_results.classification_goal_fu
nction_result.ClassificationGoalFunctionResult.get_colored_output
[317]: textattack.goal_function_results.html#textattack.goal_function_results.classification_goal_fu
nction_result.ClassificationGoalFunctionResult.get_text_color_input
[318]: textattack.goal_function_results.html#textattack.goal_function_results.classification_goal_fu
nction_result.ClassificationGoalFunctionResult.get_text_color_perturbed
[319]: textattack.goal_function_results.html#goalfunctionresult-class
[320]: textattack.goal_function_results.html#textattack.goal_function_results.goal_function_result.G
oalFunctionResult
[321]: textattack.goal_function_results.html#textattack.goal_function_results.goal_function_result.G
oalFunctionResult.get_colored_output
[322]: textattack.goal_function_results.html#textattack.goal_function_results.goal_function_result.G
oalFunctionResult.get_text_color_input
[323]: textattack.goal_function_results.html#textattack.goal_function_results.goal_function_result.G
oalFunctionResult.get_text_color_perturbed
[324]: textattack.goal_function_results.html#textattack.goal_function_results.goal_function_result.G
oalFunctionResultStatus
[325]: textattack.goal_function_results.html#textattack.goal_function_results.goal_function_result.G
oalFunctionResultStatus.MAXIMIZING
[326]: textattack.goal_function_results.html#textattack.goal_function_results.goal_function_result.G
oalFunctionResultStatus.SEARCHING
[327]: textattack.goal_function_results.html#textattack.goal_function_results.goal_function_result.G
oalFunctionResultStatus.SKIPPED
[328]: textattack.goal_function_results.html#textattack.goal_function_results.goal_function_result.G
oalFunctionResultStatus.SUCCEEDED
[329]: textattack.goal_function_results.html#texttotextgoalfunctionresult-class
[330]: textattack.goal_function_results.html#textattack.goal_function_results.text_to_text_goal_func
tion_result.TextToTextGoalFunctionResult
[331]: textattack.goal_function_results.html#textattack.goal_function_results.text_to_text_goal_func
tion_result.TextToTextGoalFunctionResult.get_colored_output
[332]: textattack.goal_function_results.html#textattack.goal_function_results.text_to_text_goal_func
tion_result.TextToTextGoalFunctionResult.get_text_color_input
[333]: textattack.goal_function_results.html#textattack.goal_function_results.text_to_text_goal_func
tion_result.TextToTextGoalFunctionResult.get_text_color_perturbed
[334]: textattack.goal_functions.html
[335]: textattack.goal_functions.html#goal-functions
[336]: textattack.goal_functions.html#subpackages
[337]: textattack.goal_functions.classification.html
[338]: textattack.goal_functions.classification.html#goal-fucntion-for-classification
[339]: textattack.goal_functions.classification.html#module-textattack.goal_functions.classification
.classification_goal_function
[340]: textattack.goal_functions.classification.html#determine-for-if-an-attack-has-been-successful-
in-classification
[341]: textattack.goal_functions.classification.html#textattack.goal_functions.classification.classi
fication_goal_function.ClassificationGoalFunction
[342]: textattack.goal_functions.classification.html#textattack.goal_functions.classification.classi
fication_goal_function.ClassificationGoalFunction.extra_repr_keys
[343]: textattack.goal_functions.classification.html#determine-if-an-attack-has-been-successful-in-h
ard-label-classficiation
[344]: textattack.goal_functions.classification.html#textattack.goal_functions.classification.hardla
bel_classification.HardLabelClassification
[345]: textattack.goal_functions.classification.html#determine-if-maintaining-the-same-predicted-lab
el-input-reduction
[346]: textattack.goal_functions.classification.html#textattack.goal_functions.classification.input_
reduction.InputReduction
[347]: textattack.goal_functions.classification.html#textattack.goal_functions.classification.input_
reduction.InputReduction.extra_repr_keys
[348]: textattack.goal_functions.classification.html#determine-if-an-attack-has-been-successful-in-t
argeted-classification
[349]: textattack.goal_functions.classification.html#textattack.goal_functions.classification.target
ed_classification.TargetedClassification
[350]: textattack.goal_functions.classification.html#textattack.goal_functions.classification.target
ed_classification.TargetedClassification.extra_repr_keys
[351]: textattack.goal_functions.classification.html#determine-successful-in-untargeted-classificati
on
[352]: textattack.goal_functions.classification.html#textattack.goal_functions.classification.untarg
eted_classification.UntargetedClassification
[353]: textattack.goal_functions.custom.html
[354]: textattack.goal_functions.custom.html#custom-goal-functions
[355]: textattack.goal_functions.custom.html#module-textattack.goal_functions.custom.logit_sum
[356]: textattack.goal_functions.custom.html#goal-function-for-logit-sum
[357]: textattack.goal_functions.custom.html#textattack.goal_functions.custom.logit_sum.LogitSum
[358]: textattack.goal_functions.custom.html#goal-function-for-namedentityrecognition
[359]: textattack.goal_functions.custom.html#textattack.goal_functions.custom.named_entity_recogniti
on.NamedEntityRecognition
[360]: textattack.goal_functions.custom.html#goal-function-for-targeted-classification-with-bonus-sc
ore
[361]: textattack.goal_functions.custom.html#textattack.goal_functions.custom.targeted_bonus.Targete
dBonus
[362]: textattack.goal_functions.custom.html#goal-function-for-strict-targeted-classification
[363]: textattack.goal_functions.custom.html#textattack.goal_functions.custom.targeted_strict.Target
edStrict
[364]: textattack.goal_functions.text.html
[365]: textattack.goal_functions.text.html#goal-function-for-text-to-text-case
[366]: textattack.goal_functions.text.html#module-textattack.goal_functions.text.maximize_levenshtei
n
[367]: textattack.goal_functions.text.html#textattack.goal_functions.text.maximize_levenshtein.Maxim
izeLevenshtein
[368]: textattack.goal_functions.text.html#textattack.goal_functions.text.maximize_levenshtein.Maxim
izeLevenshtein.clear_cache
[369]: textattack.goal_functions.text.html#textattack.goal_functions.text.maximize_levenshtein.Maxim
izeLevenshtein.extra_repr_keys
[370]: textattack.goal_functions.text.html#goal-function-for-attempts-to-minimize-the-bleu-score
[371]: textattack.goal_functions.text.html#textattack.goal_functions.text.minimize_bleu.MinimizeBleu
[372]: textattack.goal_functions.text.html#textattack.goal_functions.text.minimize_bleu.MinimizeBleu
.clear_cache
[373]: textattack.goal_functions.text.html#textattack.goal_functions.text.minimize_bleu.MinimizeBleu
.extra_repr_keys
[374]: textattack.goal_functions.text.html#textattack.goal_functions.text.minimize_bleu.MinimizeBleu
.EPS
[375]: textattack.goal_functions.text.html#textattack.goal_functions.text.minimize_bleu.get_bleu
[376]: textattack.goal_functions.text.html#goal-function-for-seq2sick
[377]: textattack.goal_functions.text.html#textattack.goal_functions.text.non_overlapping_output.Non
OverlappingOutput
[378]: textattack.goal_functions.text.html#textattack.goal_functions.text.non_overlapping_output.Non
OverlappingOutput.clear_cache
[379]: textattack.goal_functions.text.html#textattack.goal_functions.text.non_overlapping_output.get
_words_cached
[380]: textattack.goal_functions.text.html#textattack.goal_functions.text.non_overlapping_output.wor
d_difference_score
[381]: textattack.goal_functions.text.html#goal-function-for-texttotext
[382]: textattack.goal_functions.text.html#textattack.goal_functions.text.text_to_text_goal_function
.TextToTextGoalFunction
[383]: textattack.goal_functions.html#module-textattack.goal_functions.goal_function
[384]: textattack.goal_functions.html#goalfunction-class
[385]: textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction
[386]: textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.clear_cac
he
[387]: textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.extra_rep
r_keys
[388]: textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.get_outpu
t
[389]: textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.get_resul
t
[390]: textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.get_resul
ts
[391]: textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction.init_atta
ck_example
[392]: textattack.llms.html
[393]: textattack.llms.html#large-language-models
[394]: textattack.llms.html#module-textattack.llms.chat_gpt_wrapper
[395]: textattack.llms.html#textattack.llms.chat_gpt_wrapper.ChatGptWrapper
[396]: textattack.llms.html#textattack.llms.huggingface_llm_wrapper.HuggingFaceLLMWrapper
[397]: textattack.loggers.html
[398]: textattack.loggers.html#misc-loggers-loggers-track-visualize-and-export-attack-results
[399]: textattack.loggers.html#module-textattack.loggers.attack_log_manager
[400]: textattack.loggers.html#managing-attack-logs
[401]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager
[402]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.add_output_csv
[403]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.add_output_fil
e
[404]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.add_output_sum
mary_json
[405]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.disable_color
[406]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.enable_stdout
[407]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.enable_visdom
[408]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.enable_wandb
[409]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.flush
[410]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_attack_det
ails
[411]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_result
[412]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_results
[413]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_sep
[414]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_summary
[415]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.log_summary_ro
ws
[416]: textattack.loggers.html#textattack.loggers.attack_log_manager.AttackLogManager.metrics
[417]: textattack.loggers.html#attack-logs-to-csv
[418]: textattack.loggers.html#textattack.loggers.csv_logger.CSVLogger
[419]: textattack.loggers.html#textattack.loggers.csv_logger.CSVLogger.close
[420]: textattack.loggers.html#textattack.loggers.csv_logger.CSVLogger.flush
[421]: textattack.loggers.html#textattack.loggers.csv_logger.CSVLogger.log_attack_result
[422]: textattack.loggers.html#attack-logs-to-file
[423]: textattack.loggers.html#textattack.loggers.file_logger.FileLogger
[424]: textattack.loggers.html#textattack.loggers.file_logger.FileLogger.close
[425]: textattack.loggers.html#textattack.loggers.file_logger.FileLogger.flush
[426]: textattack.loggers.html#textattack.loggers.file_logger.FileLogger.log_attack_result
[427]: textattack.loggers.html#textattack.loggers.file_logger.FileLogger.log_sep
[428]: textattack.loggers.html#textattack.loggers.file_logger.FileLogger.log_summary_rows
[429]: textattack.loggers.html#attack-summary-results-logs-to-json
[430]: textattack.loggers.html#textattack.loggers.json_summary_logger.JsonSummaryLogger
[431]: textattack.loggers.html#textattack.loggers.json_summary_logger.JsonSummaryLogger.close
[432]: textattack.loggers.html#textattack.loggers.json_summary_logger.JsonSummaryLogger.flush
[433]: textattack.loggers.html#textattack.loggers.json_summary_logger.JsonSummaryLogger.log_summary_
rows
[434]: textattack.loggers.html#attack-logger-wrapper
[435]: textattack.loggers.html#textattack.loggers.logger.Logger
[436]: textattack.loggers.html#textattack.loggers.logger.Logger.close
[437]: textattack.loggers.html#textattack.loggers.logger.Logger.flush
[438]: textattack.loggers.html#textattack.loggers.logger.Logger.log_attack_result
[439]: textattack.loggers.html#textattack.loggers.logger.Logger.log_hist
[440]: textattack.loggers.html#textattack.loggers.logger.Logger.log_sep
[441]: textattack.loggers.html#textattack.loggers.logger.Logger.log_summary_rows
[442]: textattack.loggers.html#attack-logs-to-visdom
[443]: textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger
[444]: textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.bar
[445]: textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.flush
[446]: textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.hist
[447]: textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.log_attack_result
[448]: textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.log_hist
[449]: textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.log_summary_rows
[450]: textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.table
[451]: textattack.loggers.html#textattack.loggers.visdom_logger.VisdomLogger.text
[452]: textattack.loggers.html#textattack.loggers.visdom_logger.port_is_open
[453]: textattack.loggers.html#attack-logs-to-wandb
[454]: textattack.loggers.html#textattack.loggers.weights_and_biases_logger.WeightsAndBiasesLogger
[455]: textattack.loggers.html#textattack.loggers.weights_and_biases_logger.WeightsAndBiasesLogger.l
og_attack_result
[456]: textattack.loggers.html#textattack.loggers.weights_and_biases_logger.WeightsAndBiasesLogger.l
og_sep
[457]: textattack.loggers.html#textattack.loggers.weights_and_biases_logger.WeightsAndBiasesLogger.l
og_summary_rows
[458]: textattack.metrics.html
[459]: textattack.metrics.html#metrics-package-to-calculate-advanced-metrics-for-evaluting-attacks-a
nd-augmented-text
[460]: textattack.metrics.html#subpackages
[461]: textattack.metrics.attack_metrics.html
[462]: textattack.metrics.attack_metrics.html#attack-metrics-package
[463]: textattack.metrics.attack_metrics.html#module-textattack.metrics.attack_metrics.attack_querie
s
[464]: textattack.metrics.attack_metrics.html#metrics-on-attackqueries
[465]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_queries.Attac
kQueries
[466]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_queries.Attac
kQueries.avg_num_queries
[467]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_queries.Attac
kQueries.calculate
[468]: textattack.metrics.attack_metrics.html#metrics-on-attacksuccessrate
[469]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_success_rate.
AttackSuccessRate
[470]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_success_rate.
AttackSuccessRate.attack_accuracy_perc
[471]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_success_rate.
AttackSuccessRate.attack_success_rate_perc
[472]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_success_rate.
AttackSuccessRate.calculate
[473]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.attack_success_rate.
AttackSuccessRate.original_accuracy_perc
[474]: textattack.metrics.attack_metrics.html#metrics-on-perturbed-words
[475]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.words_perturbed.Word
sPerturbed
[476]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.words_perturbed.Word
sPerturbed.avg_number_word_perturbed_num
[477]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.words_perturbed.Word
sPerturbed.avg_perturbation_perc
[478]: textattack.metrics.attack_metrics.html#textattack.metrics.attack_metrics.words_perturbed.Word
sPerturbed.calculate
[479]: textattack.metrics.quality_metrics.html
[480]: textattack.metrics.quality_metrics.html#metrics-on-quality-package
[481]: textattack.metrics.quality_metrics.html#module-textattack.metrics.quality_metrics.bert_score
[482]: textattack.metrics.quality_metrics.html#bertscoremetric-class
[483]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.bert_score.BERTSco
reMetric
[484]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.bert_score.BERTSco
reMetric.calculate
[485]: textattack.metrics.quality_metrics.html#meteormetric-class
[486]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.meteor_score.Meteo
rMetric
[487]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.meteor_score.Meteo
rMetric.calculate
[488]: textattack.metrics.quality_metrics.html#perplexity-metric
[489]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.perplexity.Perplex
ity
[490]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.perplexity.Perplex
ity.calc_ppl
[491]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.perplexity.Perplex
ity.calculate
[492]: textattack.metrics.quality_metrics.html#usemetric-class
[493]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.sentence_bert.SBER
TMetric
[494]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.sentence_bert.SBER
TMetric.calculate
[495]: textattack.metrics.quality_metrics.html#id1
[496]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.use.USEMetric
[497]: textattack.metrics.quality_metrics.html#textattack.metrics.quality_metrics.use.USEMetric.calc
ulate
[498]: textattack.metrics.html#module-textattack.metrics.metric
[499]: textattack.metrics.html#metric-class
[500]: textattack.metrics.html#textattack.metrics.metric.Metric
[501]: textattack.metrics.html#textattack.metrics.metric.Metric.calculate
[502]: textattack.metrics.html#attack-metric-quality-recipes
[503]: textattack.metrics.html#textattack.metrics.recipe.AdvancedAttackMetric
[504]: textattack.metrics.html#textattack.metrics.recipe.AdvancedAttackMetric.calculate
[505]: textattack.models.html
[506]: textattack.models.html#models
[507]: textattack.models.html#models-user-specified
[508]: textattack.models.html#models-pre-trained
[509]: textattack.models.html#model-wrappers
[510]: textattack.models.html#subpackages
[511]: textattack.models.helpers.html
[512]: textattack.models.helpers.html#moderl-helpers
[513]: textattack.models.helpers.html#module-textattack.models.helpers.glove_embedding_layer
[514]: textattack.models.helpers.html#glove-embedding
[515]: textattack.models.helpers.html#textattack.models.helpers.glove_embedding_layer.EmbeddingLayer
[516]: textattack.models.helpers.html#textattack.models.helpers.glove_embedding_layer.EmbeddingLayer
.forward
[517]: textattack.models.helpers.html#textattack.models.helpers.glove_embedding_layer.EmbeddingLayer
.training
[518]: textattack.models.helpers.html#textattack.models.helpers.glove_embedding_layer.GloveEmbedding
Layer
[519]: textattack.models.helpers.html#textattack.models.helpers.glove_embedding_layer.GloveEmbedding
Layer.EMBEDDING_PATH
[520]: textattack.models.helpers.html#textattack.models.helpers.glove_embedding_layer.GloveEmbedding
Layer.training
[521]: textattack.models.helpers.html#lstm-4-classification
[522]: textattack.models.helpers.html#textattack.models.helpers.lstm_for_classification.LSTMForClass
ification
[523]: textattack.models.helpers.html#textattack.models.helpers.lstm_for_classification.LSTMForClass
ification.forward
[524]: textattack.models.helpers.html#textattack.models.helpers.lstm_for_classification.LSTMForClass
ification.from_pretrained
[525]: textattack.models.helpers.html#textattack.models.helpers.lstm_for_classification.LSTMForClass
ification.get_input_embeddings
[526]: textattack.models.helpers.html#textattack.models.helpers.lstm_for_classification.LSTMForClass
ification.load_from_disk
[527]: textattack.models.helpers.html#textattack.models.helpers.lstm_for_classification.LSTMForClass
ification.save_pretrained
[528]: textattack.models.helpers.html#textattack.models.helpers.lstm_for_classification.LSTMForClass
ification.training
[529]: textattack.models.helpers.html#t5-model-trained-to-generate-text-from-text
[530]: textattack.models.helpers.html#textattack.models.helpers.t5_for_text_to_text.T5ForTextToText
[531]: textattack.models.helpers.html#textattack.models.helpers.t5_for_text_to_text.T5ForTextToText.
from_pretrained
[532]: textattack.models.helpers.html#textattack.models.helpers.t5_for_text_to_text.T5ForTextToText.
get_input_embeddings
[533]: textattack.models.helpers.html#textattack.models.helpers.t5_for_text_to_text.T5ForTextToText.
save_pretrained
[534]: textattack.models.helpers.html#textattack.models.helpers.t5_for_text_to_text.T5ForTextToText.
training
[535]: textattack.models.helpers.html#util-function-for-model-wrapper
[536]: textattack.models.helpers.html#textattack.models.helpers.utils.load_cached_state_dict
[537]: textattack.models.helpers.html#word-cnn-for-classification
[538]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.CNNTextL
ayer
[539]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.CNNTextL
ayer.forward
[540]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.CNNTextL
ayer.training
[541]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.WordCNNF
orClassification
[542]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.WordCNNF
orClassification.forward
[543]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.WordCNNF
orClassification.from_pretrained
[544]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.WordCNNF
orClassification.get_input_embeddings
[545]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.WordCNNF
orClassification.load_from_disk
[546]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.WordCNNF
orClassification.save_pretrained
[547]: textattack.models.helpers.html#textattack.models.helpers.word_cnn_for_classification.WordCNNF
orClassification.training
[548]: textattack.models.tokenizers.html
[549]: textattack.models.tokenizers.html#tokenizers-for-model-wrapper
[550]: textattack.models.tokenizers.html#module-textattack.models.tokenizers.glove_tokenizer
[551]: textattack.models.tokenizers.html#glove-tokenizer
[552]: textattack.models.tokenizers.html#textattack.models.tokenizers.glove_tokenizer.GloveTokenizer
[553]: textattack.models.tokenizers.html#textattack.models.tokenizers.glove_tokenizer.GloveTokenizer
.batch_encode
[554]: textattack.models.tokenizers.html#textattack.models.tokenizers.glove_tokenizer.GloveTokenizer
.convert_ids_to_tokens
[555]: textattack.models.tokenizers.html#textattack.models.tokenizers.glove_tokenizer.GloveTokenizer
.encode
[556]: textattack.models.tokenizers.html#textattack.models.tokenizers.glove_tokenizer.WordLevelToken
izer
[557]: textattack.models.tokenizers.html#t5-tokenizer
[558]: textattack.models.tokenizers.html#textattack.models.tokenizers.t5_tokenizer.T5Tokenizer
[559]: textattack.models.tokenizers.html#textattack.models.tokenizers.t5_tokenizer.T5Tokenizer.decod
e
[560]: textattack.models.wrappers.html
[561]: textattack.models.wrappers.html#model-wrappers-package
[562]: textattack.models.wrappers.html#module-textattack.models.wrappers.huggingface_model_wrapper
[563]: textattack.models.wrappers.html#huggingface-model-wrapper
[564]: textattack.models.wrappers.html#textattack.models.wrappers.huggingface_model_wrapper.HuggingF
aceModelWrapper
[565]: textattack.models.wrappers.html#textattack.models.wrappers.huggingface_model_wrapper.HuggingF
aceModelWrapper.get_grad
[566]: textattack.models.wrappers.html#modelwrapper-class
[567]: textattack.models.wrappers.html#textattack.models.wrappers.model_wrapper.ModelWrapper
[568]: textattack.models.wrappers.html#textattack.models.wrappers.model_wrapper.ModelWrapper.get_gra
d
[569]: textattack.models.wrappers.html#textattack.models.wrappers.model_wrapper.ModelWrapper.tokeniz
e
[570]: textattack.models.wrappers.html#pytorch-model-wrapper
[571]: textattack.models.wrappers.html#textattack.models.wrappers.pytorch_model_wrapper.PyTorchModel
Wrapper
[572]: textattack.models.wrappers.html#textattack.models.wrappers.pytorch_model_wrapper.PyTorchModel
Wrapper.get_grad
[573]: textattack.models.wrappers.html#textattack.models.wrappers.pytorch_model_wrapper.PyTorchModel
Wrapper.to
[574]: textattack.models.wrappers.html#scikit-learn-model-wrapper
[575]: textattack.models.wrappers.html#textattack.models.wrappers.sklearn_model_wrapper.SklearnModel
Wrapper
[576]: textattack.models.wrappers.html#textattack.models.wrappers.sklearn_model_wrapper.SklearnModel
Wrapper.get_grad
[577]: textattack.models.wrappers.html#tensorflow-model-wrapper
[578]: textattack.models.wrappers.html#textattack.models.wrappers.tensorflow_model_wrapper.TensorFlo
wModelWrapper
[579]: textattack.models.wrappers.html#textattack.models.wrappers.tensorflow_model_wrapper.TensorFlo
wModelWrapper.get_grad
[580]: textattack.prompt_augmentation.html
[581]: textattack.prompt_augmentation.html#prompt-augmentation
[582]: textattack.prompt_augmentation.html#module-textattack.prompt_augmentation.prompt_augmentation
_pipeline
[583]: textattack.prompt_augmentation.html#textattack.prompt_augmentation.prompt_augmentation_pipeli
ne.PromptAugmentationPipeline
[584]: textattack.search_methods.html
[585]: textattack.search_methods.html#search-methods
[586]: textattack.search_methods.html#module-textattack.search_methods.alzantot_genetic_algorithm
[587]: textattack.search_methods.html#reimplementation-of-search-method-from-generating-natural-lang
uage-adversarial-examples
[588]: textattack.search_methods.html#textattack.search_methods.alzantot_genetic_algorithm.AlzantotG
eneticAlgorithm
[589]: textattack.search_methods.html#beam-search
[590]: textattack.search_methods.html#textattack.search_methods.beam_search.BeamSearch
[591]: textattack.search_methods.html#textattack.search_methods.beam_search.BeamSearch.extra_repr_ke
ys
[592]: textattack.search_methods.html#textattack.search_methods.beam_search.BeamSearch.perform_searc
h
[593]: textattack.search_methods.html#textattack.search_methods.beam_search.BeamSearch.is_black_box
[594]: textattack.search_methods.html#textattack.search_methods.differential_evolution.DifferentialE
volution
[595]: textattack.search_methods.html#textattack.search_methods.differential_evolution.DifferentialE
volution.check_transformation_compatibility
[596]: textattack.search_methods.html#textattack.search_methods.differential_evolution.DifferentialE
volution.extra_repr_keys
[597]: textattack.search_methods.html#textattack.search_methods.differential_evolution.DifferentialE
volution.perform_search
[598]: textattack.search_methods.html#textattack.search_methods.differential_evolution.DifferentialE
volution.is_black_box
[599]: textattack.search_methods.html#genetic-algorithm-word-swap
[600]: textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgorithm
[601]: textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgorithm.c
heck_transformation_compatibility
[602]: textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgorithm.e
xtra_repr_keys
[603]: textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgorithm.p
erform_search
[604]: textattack.search_methods.html#textattack.search_methods.genetic_algorithm.GeneticAlgorithm.i
s_black_box
[605]: textattack.search_methods.html#greedy-search
[606]: textattack.search_methods.html#textattack.search_methods.greedy_search.GreedySearch
[607]: textattack.search_methods.html#textattack.search_methods.greedy_search.GreedySearch.extra_rep
r_keys
[608]: textattack.search_methods.html#greedy-word-swap-with-word-importance-ranking
[609]: textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWordSwapW
IR
[610]: textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWordSwapW
IR.check_transformation_compatibility
[611]: textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWordSwapW
IR.extra_repr_keys
[612]: textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWordSwapW
IR.perform_search
[613]: textattack.search_methods.html#textattack.search_methods.greedy_word_swap_wir.GreedyWordSwapW
IR.is_black_box
[614]: textattack.search_methods.html#reimplementation-of-search-method-from-xiaosen-wang-hao-jin-ku
n-he-2019
[615]: textattack.search_methods.html#textattack.search_methods.improved_genetic_algorithm.ImprovedG
eneticAlgorithm
[616]: textattack.search_methods.html#textattack.search_methods.improved_genetic_algorithm.ImprovedG
eneticAlgorithm.extra_repr_keys
[617]: textattack.search_methods.html#particle-swarm-optimization
[618]: textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.Particle
SwarmOptimization
[619]: textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.Particle
SwarmOptimization.check_transformation_compatibility
[620]: textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.Particle
SwarmOptimization.extra_repr_keys
[621]: textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.Particle
SwarmOptimization.perform_search
[622]: textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.Particle
SwarmOptimization.is_black_box
[623]: textattack.search_methods.html#textattack.search_methods.particle_swarm_optimization.normaliz
e
[624]: textattack.search_methods.html#population-based-search-abstract-class
[625]: textattack.search_methods.html#textattack.search_methods.population_based_search.PopulationBa
sedSearch
[626]: textattack.search_methods.html#textattack.search_methods.population_based_search.PopulationMe
mber
[627]: textattack.search_methods.html#textattack.search_methods.population_based_search.PopulationMe
mber.num_words
[628]: textattack.search_methods.html#textattack.search_methods.population_based_search.PopulationMe
mber.score
[629]: textattack.search_methods.html#textattack.search_methods.population_based_search.PopulationMe
mber.words
[630]: textattack.search_methods.html#search-method-abstract-class
[631]: textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod
[632]: textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod.check_tra
nsformation_compatibility
[633]: textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod.get_victi
m_model
[634]: textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod.perform_s
earch
[635]: textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod.is_black_
box
[636]: textattack.shared.html
[637]: textattack.shared.html#shared-textattack-functions
[638]: textattack.shared.html#subpackages
[639]: textattack.shared.utils.html
[640]: textattack.shared.utils.html#module-textattack.shared.utils.importing
[641]: textattack.shared.utils.html#textattack.shared.utils.importing.LazyLoader
[642]: textattack.shared.utils.html#textattack.shared.utils.importing.load_module_from_file
[643]: textattack.shared.utils.html#textattack.shared.utils.install.download_from_s3
[644]: textattack.shared.utils.html#textattack.shared.utils.install.download_from_url
[645]: textattack.shared.utils.html#textattack.shared.utils.install.http_get
[646]: textattack.shared.utils.html#textattack.shared.utils.install.path_in_cache
[647]: textattack.shared.utils.html#textattack.shared.utils.install.s3_url
[648]: textattack.shared.utils.html#textattack.shared.utils.install.set_cache_dir
[649]: textattack.shared.utils.html#textattack.shared.utils.install.unzip_file
[650]: textattack.shared.utils.html#textattack.shared.utils.misc.get_textattack_model_num_labels
[651]: textattack.shared.utils.html#textattack.shared.utils.misc.hashable
[652]: textattack.shared.utils.html#textattack.shared.utils.misc.html_style_from_dict
[653]: textattack.shared.utils.html#textattack.shared.utils.misc.html_table_from_rows
[654]: textattack.shared.utils.html#textattack.shared.utils.misc.load_textattack_model_from_path
[655]: textattack.shared.utils.html#textattack.shared.utils.misc.set_seed
[656]: textattack.shared.utils.html#textattack.shared.utils.misc.sigmoid
[657]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES
[658]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.BOLD
[659]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.BROWN
[660]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.CYAN
[661]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.FAIL
[662]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.GRAY
[663]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.HEADER
[664]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.OKBLUE
[665]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.OKGREEN
[666]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.ORANGE
[667]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.PINK
[668]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.PURPLE
[669]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.STOP
[670]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.UNDERLINE
[671]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.WARNING
[672]: textattack.shared.utils.html#textattack.shared.utils.strings.ANSI_ESCAPE_CODES.YELLOW
[673]: textattack.shared.utils.html#textattack.shared.utils.strings.ReprMixin
[674]: textattack.shared.utils.html#textattack.shared.utils.strings.ReprMixin.extra_repr_keys
[675]: textattack.shared.utils.html#textattack.shared.utils.strings.TextAttackFlairTokenizer
[676]: textattack.shared.utils.html#textattack.shared.utils.strings.TextAttackFlairTokenizer.tokeniz
e
[677]: textattack.shared.utils.html#textattack.shared.utils.strings.add_indent
[678]: textattack.shared.utils.html#textattack.shared.utils.strings.check_if_punctuations
[679]: textattack.shared.utils.html#textattack.shared.utils.strings.check_if_subword
[680]: textattack.shared.utils.html#textattack.shared.utils.strings.color_from_label
[681]: textattack.shared.utils.html#textattack.shared.utils.strings.color_from_output
[682]: textattack.shared.utils.html#textattack.shared.utils.strings.color_text
[683]: textattack.shared.utils.html#textattack.shared.utils.strings.default_class_repr
[684]: textattack.shared.utils.html#textattack.shared.utils.strings.flair_tag
[685]: textattack.shared.utils.html#textattack.shared.utils.strings.has_letter
[686]: textattack.shared.utils.html#textattack.shared.utils.strings.is_one_word
[687]: textattack.shared.utils.html#textattack.shared.utils.strings.process_label_name
[688]: textattack.shared.utils.html#textattack.shared.utils.strings.strip_BPE_artifacts
[689]: textattack.shared.utils.html#textattack.shared.utils.strings.words_from_text
[690]: textattack.shared.utils.html#textattack.shared.utils.strings.zip_flair_result
[691]: textattack.shared.utils.html#textattack.shared.utils.strings.zip_stanza_result
[692]: textattack.shared.utils.html#textattack.shared.utils.tensor.batch_model_predict
[693]: textattack.shared.html#module-textattack.shared.attacked_text
[694]: textattack.shared.html#attacked-text-class
[695]: textattack.shared.html#textattack.shared.attacked_text.AttackedText
[696]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.align_with_model_tokens
[697]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.all_words_diff
[698]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.convert_from_original_idx
s
[699]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.delete_word_at_index
[700]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.first_word_diff
[701]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.first_word_diff_index
[702]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.free_memory
[703]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.generate_new_attacked_tex
t
[704]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.get_deletion_indices
[705]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.insert_text_after_word_in
dex
[706]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.insert_text_before_word_i
ndex
[707]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.ith_word_diff
[708]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.ner_of_word_index
[709]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.pos_of_word_index
[710]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.printable_text
[711]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.replace_word_at_index
[712]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.replace_words_at_indices
[713]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.text_after_word_index
[714]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.text_until_word_index
[715]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.text_window_around_index
[716]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.words_diff_num
[717]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.words_diff_ratio
[718]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.SPLIT_TOKEN
[719]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.column_labels
[720]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.newly_swapped_words
[721]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.num_words
[722]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.text
[723]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.tokenizer_input
[724]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.words
[725]: textattack.shared.html#textattack.shared.attacked_text.AttackedText.words_per_input
[726]: textattack.shared.html#misc-checkpoints
[727]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint
[728]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.load
[729]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.save
[730]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.dataset_offset
[731]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.datetime
[732]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_failed_attacks
[733]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_maximized_attacks
[734]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_remaining_attacks
[735]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_skipped_attacks
[736]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.num_successful_attacks
[737]: textattack.shared.html#textattack.shared.checkpoint.AttackCheckpoint.results_count
[738]: textattack.shared.html#shared-data-fields
[739]: textattack.shared.html#misc-validators
[740]: textattack.shared.html#textattack.shared.validators.transformation_consists_of
[741]: textattack.shared.html#textattack.shared.validators.transformation_consists_of_word_swaps
[742]: textattack.shared.html#textattack.shared.validators.transformation_consists_of_word_swaps_and
_deletions
[743]: textattack.shared.html#textattack.shared.validators.transformation_consists_of_word_swaps_dif
ferential_evolution
[744]: textattack.shared.html#textattack.shared.validators.validate_model_goal_function_compatibilit
y
[745]: textattack.shared.html#textattack.shared.validators.validate_model_gradient_word_swap_compati
bility
[746]: textattack.shared.html#shared-loads-word-embeddings-and-related-distances
[747]: textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding
[748]: textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.get_cos_sim
[749]: textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.get_mse_dist
[750]: textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.index2word
[751]: textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.nearest_neighb
ours
[752]: textattack.shared.html#textattack.shared.word_embeddings.AbstractWordEmbedding.word2index
[753]: textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding
[754]: textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.get_cos_sim
[755]: textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.get_mse_dist
[756]: textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.index2word
[757]: textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.nearest_neighbou
rs
[758]: textattack.shared.html#textattack.shared.word_embeddings.GensimWordEmbedding.word2index
[759]: textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding
[760]: textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.counterfitted_GLOVE_em
bedding
[761]: textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.get_cos_sim
[762]: textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.get_mse_dist
[763]: textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.index2word
[764]: textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.nearest_neighbours
[765]: textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.word2index
[766]: textattack.shared.html#textattack.shared.word_embeddings.WordEmbedding.PATH
[767]: textattack.transformations.html
[768]: textattack.transformations.html#transformations
[769]: textattack.transformations.html#subpackages
[770]: textattack.transformations.sentence_transformations.html
[771]: textattack.transformations.sentence_transformations.html#sentence-transformations-package
[772]: textattack.transformations.sentence_transformations.html#module-textattack.transformations.se
ntence_transformations.back_transcription
[773]: textattack.transformations.sentence_transformations.html#backtranscription-class
[774]: textattack.transformations.sentence_transformations.html#textattack.transformations.sentence_
transformations.back_transcription.BackTranscription
[775]: textattack.transformations.sentence_transformations.html#textattack.transformations.sentence_
transformations.back_transcription.BackTranscription.back_transcribe
[776]: textattack.transformations.sentence_transformations.html#backtranslation-class
[777]: textattack.transformations.sentence_transformations.html#textattack.transformations.sentence_
transformations.back_translation.BackTranslation
[778]: textattack.transformations.sentence_transformations.html#textattack.transformations.sentence_
transformations.back_translation.BackTranslation.translate
[779]: textattack.transformations.sentence_transformations.html#sentencetransformation-class
[780]: textattack.transformations.sentence_transformations.html#textattack.transformations.sentence_
transformations.sentence_transformation.SentenceTransformation
[781]: textattack.transformations.word_insertions.html
[782]: textattack.transformations.word_insertions.html#word-insertions-package
[783]: textattack.transformations.word_insertions.html#module-textattack.transformations.word_insert
ions.word_insertion
[784]: textattack.transformations.word_insertions.html#wordinsertion-class
[785]: textattack.transformations.word_insertions.html#textattack.transformations.word_insertions.wo
rd_insertion.WordInsertion
[786]: textattack.transformations.word_insertions.html#wordinsertionmaskedlm-class
[787]: textattack.transformations.word_insertions.html#textattack.transformations.word_insertions.wo
rd_insertion_masked_lm.WordInsertionMaskedLM
[788]: textattack.transformations.word_insertions.html#textattack.transformations.word_insertions.wo
rd_insertion_masked_lm.WordInsertionMaskedLM.extra_repr_keys
[789]: textattack.transformations.word_insertions.html#wordinsertionrandomsynonym-class
[790]: textattack.transformations.word_insertions.html#textattack.transformations.word_insertions.wo
rd_insertion_random_synonym.WordInsertionRandomSynonym
[791]: textattack.transformations.word_insertions.html#textattack.transformations.word_insertions.wo
rd_insertion_random_synonym.WordInsertionRandomSynonym.deterministic
[792]: textattack.transformations.word_insertions.html#textattack.transformations.word_insertions.wo
rd_insertion_random_synonym.check_if_one_word
[793]: textattack.transformations.word_merges.html
[794]: textattack.transformations.word_merges.html#word-merges-package
[795]: textattack.transformations.word_merges.html#module-textattack.transformations.word_merges.wor
d_merge
[796]: textattack.transformations.word_merges.html#word-merge
[797]: textattack.transformations.word_merges.html#textattack.transformations.word_merges.word_merge
.WordMerge
[798]: textattack.transformations.word_merges.html#wordmergemaskedlm-class
[799]: textattack.transformations.word_merges.html#textattack.transformations.word_merges.word_merge
_masked_lm.WordMergeMaskedLM
[800]: textattack.transformations.word_merges.html#textattack.transformations.word_merges.word_merge
_masked_lm.WordMergeMaskedLM.extra_repr_keys
[801]: textattack.transformations.word_merges.html#textattack.transformations.word_merges.word_merge
_masked_lm.find_merge_index
[802]: textattack.transformations.word_swaps.html
[803]: textattack.transformations.word_swaps.html#word-swaps-package
[804]: textattack.transformations.word_swaps.html#subpackages
[805]: textattack.transformations.word_swaps.chn_transformations.html
[806]: textattack.transformations.word_swaps.chn_transformations.html#chinese-transformations-packag
e
[807]: textattack.transformations.word_swaps.chn_transformations.html#module-textattack.transformati
ons.word_swaps.chn_transformations.chinese_homophone_character_swap
[808]: textattack.transformations.word_swaps.html#module-textattack.transformations.word_swaps.word_
swap
[809]: textattack.transformations.word_swaps.html#word-swap
[810]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap.Wo
rdSwap
[811]: textattack.transformations.word_swaps.html#word-swap-by-changing-location
[812]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ch
ange_location.WordSwapChangeLocation
[813]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ch
ange_location.idx_to_words
[814]: textattack.transformations.word_swaps.html#word-swap-by-changing-name
[815]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ch
ange_name.WordSwapChangeName
[816]: textattack.transformations.word_swaps.html#word-swap-by-changing-number
[817]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ch
ange_number.WordSwapChangeNumber
[818]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ch
ange_number.idx_to_words
[819]: textattack.transformations.word_swaps.html#word-swap-by-contraction
[820]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_co
ntract.WordSwapContract
[821]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_co
ntract.WordSwapContract.reverse_contraction_map
[822]: textattack.transformations.word_swaps.html#word-swap-by-invisible-deletions
[823]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_de
letions.WordSwapDeletions
[824]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_de
letions.WordSwapDeletions.apply_perturbation
[825]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_de
letions.WordSwapDeletions.extra_repr_keys
[826]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_de
letions.WordSwapDeletions.deterministic
[827]: textattack.transformations.word_swaps.html#word-swap-for-differential-evolution
[828]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_di
fferential_evolution.WordSwapDifferentialEvolution
[829]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_di
fferential_evolution.WordSwapDifferentialEvolution.apply_perturbation
[830]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_di
fferential_evolution.WordSwapDifferentialEvolution.get_bounds_and_precomputed
[831]: textattack.transformations.word_swaps.html#word-swap-by-embedding
[832]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_em
bedding.WordSwapEmbedding
[833]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_em
bedding.WordSwapEmbedding.extra_repr_keys
[834]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_em
bedding.recover_word_case
[835]: textattack.transformations.word_swaps.html#word-swap-by-extension
[836]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ex
tend.WordSwapExtend
[837]: textattack.transformations.word_swaps.html#word-swap-by-gradient
[838]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_gr
adient_based.WordSwapGradientBased
[839]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_gr
adient_based.WordSwapGradientBased.extra_repr_keys
[840]: textattack.transformations.word_swaps.html#word-swap-by-homoglyph
[841]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ho
moglyph_swap.WordSwapHomoglyphSwap
[842]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ho
moglyph_swap.WordSwapHomoglyphSwap.apply_perturbation
[843]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ho
moglyph_swap.WordSwapHomoglyphSwap.extra_repr_keys
[844]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ho
moglyph_swap.WordSwapHomoglyphSwap.deterministic
[845]: textattack.transformations.word_swaps.html#word-swap-by-openhownet
[846]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ho
wnet.WordSwapHowNet
[847]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ho
wnet.WordSwapHowNet.extra_repr_keys
[848]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ho
wnet.WordSwapHowNet.PATH
[849]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ho
wnet.recover_word_case
[850]: textattack.transformations.word_swaps.html#word-swap-by-inflections
[851]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_in
flections.WordSwapInflections
[852]: textattack.transformations.word_swaps.html#word-swap-by-invisible-characters
[853]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_in
visible_characters.WordSwapInvisibleCharacters
[854]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_in
visible_characters.WordSwapInvisibleCharacters.apply_perturbation
[855]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_in
visible_characters.WordSwapInvisibleCharacters.extra_repr_keys
[856]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_in
visible_characters.WordSwapInvisibleCharacters.deterministic
[857]: textattack.transformations.word_swaps.html#word-swap-by-bert-masked-lm
[858]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ma
sked_lm.WordSwapMaskedLM
[859]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ma
sked_lm.WordSwapMaskedLM.extra_repr_keys
[860]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ma
sked_lm.recover_word_case
[861]: textattack.transformations.word_swaps.html#word-swap-by-neighboring-character-swap
[862]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ne
ighboring_character_swap.WordSwapNeighboringCharacterSwap
[863]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ne
ighboring_character_swap.WordSwapNeighboringCharacterSwap.extra_repr_keys
[864]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ne
ighboring_character_swap.WordSwapNeighboringCharacterSwap.deterministic
[865]: textattack.transformations.word_swaps.html#word-swap-by-swaps-characters-with-qwerty-adjacent
-keys
[866]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_qw
erty.WordSwapQWERTY
[867]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_qw
erty.WordSwapQWERTY.deterministic
[868]: textattack.transformations.word_swaps.html#word-swap-by-random-character-deletion
[869]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ra
ndom_character_deletion.WordSwapRandomCharacterDeletion
[870]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ra
ndom_character_deletion.WordSwapRandomCharacterDeletion.extra_repr_keys
[871]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ra
ndom_character_deletion.WordSwapRandomCharacterDeletion.deterministic
[872]: textattack.transformations.word_swaps.html#word-swap-by-random-character-insertion
[873]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ra
ndom_character_insertion.WordSwapRandomCharacterInsertion
[874]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ra
ndom_character_insertion.WordSwapRandomCharacterInsertion.extra_repr_keys
[875]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ra
ndom_character_insertion.WordSwapRandomCharacterInsertion.deterministic
[876]: textattack.transformations.word_swaps.html#word-swap-by-random-character-substitution
[877]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ra
ndom_character_substitution.WordSwapRandomCharacterSubstitution
[878]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ra
ndom_character_substitution.WordSwapRandomCharacterSubstitution.extra_repr_keys
[879]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_ra
ndom_character_substitution.WordSwapRandomCharacterSubstitution.deterministic
[880]: textattack.transformations.word_swaps.html#word-swap-by-invisible-reorderings
[881]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_re
orderings.WordSwapReorderings
[882]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_re
orderings.WordSwapReorderings.apply_perturbation
[883]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_re
orderings.WordSwapReorderings.extra_repr_keys
[884]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_re
orderings.WordSwapReorderings.deterministic
[885]: textattack.transformations.word_swaps.html#word-swap-by-swapping-synonyms-in-wordnet
[886]: textattack.transformations.word_swaps.html#textattack.transformations.word_swaps.word_swap_wo
rdnet.WordSwapWordNet
[887]: textattack.transformations.html#module-textattack.transformations.composite_transformation
[888]: textattack.transformations.html#composite-transformation
[889]: textattack.transformations.html#textattack.transformations.composite_transformation.Composite
Transformation
[890]: textattack.transformations.html#transformation-abstract-class
[891]: textattack.transformations.html#textattack.transformations.transformation.Transformation
[892]: textattack.transformations.html#textattack.transformations.transformation.Transformation.dete
rministic
[893]: textattack.transformations.html#word-deletion-transformation
[894]: textattack.transformations.html#textattack.transformations.word_deletion.WordDeletion
[895]: textattack.transformations.html#word-swap-transformation-by-swapping-the-order-of-words
[896]: textattack.transformations.html#textattack.transformations.word_innerswap_random.WordInnerSwa
pRandom
[897]: textattack.transformations.html#textattack.transformations.word_innerswap_random.WordInnerSwa
pRandom.deterministic
[898]: #module-textattack.attack
[899]: #attack-class
[900]: textattack.goal_functions.html#textattack.goal_functions.goal_function.GoalFunction
[901]: textattack.constraints.html#textattack.constraints.constraint.Constraint
[902]: textattack.constraints.html#textattack.constraints.pre_transformation_constraint.PreTransform
ationConstraint
[903]: textattack.transformations.html#textattack.transformations.transformation.Transformation
[904]: textattack.search_methods.html#textattack.search_methods.search_method.SearchMethod
[905]: ../_modules/textattack/attack.html#Attack
[906]: #textattack.attack.Attack
[907]: #textattack.attack.Attack.attack
[908]: ../api/goal_functions.html#textattack.goal_functions.GoalFunction
[909]: ../api/constraints.html#textattack.constraints.Constraint
[910]: ../api/constraints.html#textattack.constraints.PreTransformationConstraint
[911]: ../api/transformations.html#textattack.transformations.Transformation
[912]: ../api/search_methods.html#textattack.search_methods.SearchMethod
[913]: ../_modules/textattack/attack.html#Attack.attack
[914]: #textattack.attack.Attack.attack
[915]: ../api/attack_results.html#textattack.attack_results.AttackResult
[916]: ../_modules/textattack/attack.html#Attack.clear_cache
[917]: #textattack.attack.Attack.clear_cache
[918]: ../_modules/textattack/attack.html#Attack.cpu_
[919]: #textattack.attack.Attack.cpu_
[920]: ../_modules/textattack/attack.html#Attack.cuda_
[921]: #textattack.attack.Attack.cuda_
[922]: ../_modules/textattack/attack.html#Attack.filter_transformations
[923]: #textattack.attack.Attack.filter_transformations
[924]: ../_modules/textattack/attack.html#Attack.get_indices_to_order
[925]: #textattack.attack.Attack.get_indices_to_order
[926]: ../_modules/textattack/attack.html#Attack.get_transformations
[927]: #textattack.attack.Attack.get_transformations
[928]: #attackargs-class
[929]: ../_modules/textattack/attack_args.html#AttackArgs
[930]: #textattack.attack_args.AttackArgs
[931]: ../api/attacker.html#textattack.Attacker
[932]: #textattack.attack_args.AttackArgs.num_examples
[933]: #textattack.attack_args.AttackArgs.num_examples
[934]: #textattack.attack_args.AttackArgs.num_successful_examples
[935]: #textattack.attack_args.AttackArgs.shuffle
[936]: ../api/goal_functions.html#textattack.goal_functions.GoalFunction
[937]: #textattack.attack_args.AttackArgs.disable_stdout
[938]: ../_modules/textattack/attack_args.html#AttackArgs.create_loggers_from_args
[939]: #textattack.attack_args.AttackArgs.create_loggers_from_args
[940]: #textattack.attack_args.AttackArgs.attack_n
[941]: #textattack.attack_args.AttackArgs.checkpoint_dir
[942]: #textattack.attack_args.AttackArgs.checkpoint_interval
[943]: #textattack.attack_args.AttackArgs.csv_coloring_style
[944]: #textattack.attack_args.AttackArgs.disable_stdout
[945]: #textattack.attack_args.AttackArgs.enable_advance_metrics
[946]: #textattack.attack_args.AttackArgs.log_summary_to_json
[947]: #textattack.attack_args.AttackArgs.log_to_csv
[948]: #textattack.attack_args.AttackArgs.log_to_txt
[949]: #textattack.attack_args.AttackArgs.log_to_visdom
[950]: #textattack.attack_args.AttackArgs.log_to_wandb
[951]: #textattack.attack_args.AttackArgs.metrics
[952]: #textattack.attack_args.AttackArgs.num_examples
[953]: #textattack.attack_args.AttackArgs.num_examples_offset
[954]: #textattack.attack_args.AttackArgs.num_successful_examples
[955]: #textattack.attack_args.AttackArgs.num_workers_per_device
[956]: #textattack.attack_args.AttackArgs.parallel
[957]: #textattack.attack_args.AttackArgs.query_budget
[958]: #textattack.attack_args.AttackArgs.random_seed
[959]: #textattack.attack_args.AttackArgs.shuffle
[960]: #textattack.attack_args.AttackArgs.silent
[961]: ../_modules/textattack/attack_args.html#CommandLineAttackArgs
[962]: #textattack.attack_args.CommandLineAttackArgs
[963]: #textattack.attack_args.AttackArgs
[964]: #textattack.dataset_args.DatasetArgs
[965]: #textattack.model_args.ModelArgs
[966]: #attacker-class
[967]: ../_modules/textattack/attacker.html#Attacker
[968]: #textattack.attacker.Attacker
[969]: ../api/attack.html#textattack.Attack
[970]: ../api/attack.html#textattack.Attack
[971]: ../api/attack.html#textattack.Attack
[972]: ../api/datasets.html#textattack.datasets.Dataset
[973]: ../api/attacker.html#textattack.AttackArgs
[974]: ../_modules/textattack/attacker.html#Attacker.attack_dataset
[975]: #textattack.attacker.Attacker.attack_dataset
[976]: ../api/attack_results.html#textattack.attack_results.AttackResult
[977]: ../_modules/textattack/attacker.html#Attacker.attack_interactive
[978]: #textattack.attacker.Attacker.attack_interactive
[979]: ../_modules/textattack/attacker.html#Attacker.from_checkpoint
[980]: #textattack.attacker.Attacker.from_checkpoint
[981]: ../api/attack.html#textattack.Attack
[982]: ../api/datasets.html#textattack.datasets.Dataset
[983]: ../_modules/textattack/attacker.html#Attacker.update_attack_args
[984]: #textattack.attacker.Attacker.update_attack_args
[985]: ../_modules/textattack/attacker.html#attack_from_queue
[986]: #textattack.attacker.attack_from_queue
[987]: ../_modules/textattack/attacker.html#pytorch_multiprocessing_workaround
[988]: #textattack.attacker.pytorch_multiprocessing_workaround
[989]: ../_modules/textattack/attacker.html#set_env_variables
[990]: #textattack.attacker.set_env_variables
[991]: #augmenterargs-class
[992]: ../_modules/textattack/augment_args.html#AugmenterArgs
[993]: #textattack.augment_args.AugmenterArgs
[994]: #textattack.augment_args.AugmenterArgs.enable_advanced_metrics
[995]: #textattack.augment_args.AugmenterArgs.exclude_original
[996]: #textattack.augment_args.AugmenterArgs.fast_augment
[997]: #textattack.augment_args.AugmenterArgs.high_yield
[998]: #textattack.augment_args.AugmenterArgs.input_column
[999]: #textattack.augment_args.AugmenterArgs.input_csv
[1000]: #textattack.augment_args.AugmenterArgs.interactive
[1001]: #textattack.augment_args.AugmenterArgs.output_csv
[1002]: #textattack.augment_args.AugmenterArgs.overwrite
[1003]: #textattack.augment_args.AugmenterArgs.pct_words_to_swap
[1004]: #textattack.augment_args.AugmenterArgs.random_seed
[1005]: #textattack.augment_args.AugmenterArgs.recipe
[1006]: #textattack.augment_args.AugmenterArgs.transformations_per_example
[1007]: #datasetargs-class
[1008]: ../_modules/textattack/dataset_args.html#DatasetArgs
[1009]: #textattack.dataset_args.DatasetArgs
[1010]: #textattack.dataset_args.DatasetArgs.dataset_by_model
[1011]: #textattack.dataset_args.DatasetArgs.dataset_from_file
[1012]: #textattack.dataset_args.DatasetArgs.dataset_from_huggingface
[1013]: #textattack.dataset_args.DatasetArgs.dataset_split
[1014]: #textattack.dataset_args.DatasetArgs.filter_by_labels
[1015]: #modelargs-class
[1016]: ../_modules/textattack/model_args.html#ModelArgs
[1017]: #textattack.model_args.ModelArgs
[1018]: #textattack.model_args.ModelArgs.model
[1019]: #textattack.model_args.ModelArgs.model_from_file
[1020]: #textattack.model_args.ModelArgs.model_from_huggingface
[1021]: #trainer-class
[1022]: ../_modules/textattack/trainer.html#Trainer
[1023]: #textattack.trainer.Trainer
[1024]: ../api/trainer.html#textattack.Trainer
[1025]: ../api/attack.html#textattack.Attack
[1026]: ../api/attack.html#textattack.Attack
[1027]: ../api/datasets.html#textattack.datasets.Dataset
[1028]: ../api/datasets.html#textattack.datasets.Dataset
[1029]: ../api/trainer.html#textattack.TrainingArgs
[1030]: ../api/trainer.html#textattack.Trainer
[1031]: ../api/trainer.html#textattack.TrainingArgs
[1032]: ../_modules/textattack/trainer.html#Trainer.evaluate
[1033]: #textattack.trainer.Trainer.evaluate
[1034]: ../_modules/textattack/trainer.html#Trainer.evaluate_step
[1035]: #textattack.trainer.Trainer.evaluate_step
[1036]: #textattack.trainer.Trainer.get_eval_dataloader
[1037]: ../_modules/textattack/trainer.html#Trainer.get_eval_dataloader
[1038]: #textattack.trainer.Trainer.get_eval_dataloader
[1039]: ../api/datasets.html#textattack.datasets.Dataset
[1040]: ../_modules/textattack/trainer.html#Trainer.get_optimizer_and_scheduler
[1041]: #textattack.trainer.Trainer.get_optimizer_and_scheduler
[1042]: ../_modules/textattack/trainer.html#Trainer.get_train_dataloader
[1043]: #textattack.trainer.Trainer.get_train_dataloader
[1044]: ../api/datasets.html#textattack.datasets.Dataset
[1045]: ../api/datasets.html#textattack.datasets.Dataset
[1046]: ../_modules/textattack/trainer.html#Trainer.train
[1047]: #textattack.trainer.Trainer.train
[1048]: ../_modules/textattack/trainer.html#Trainer.training_step
[1049]: #textattack.trainer.Trainer.training_step
[1050]: #textattack.trainer.Trainer.get_train_dataloader
[1051]: #trainingargs-class
[1052]: ../_modules/textattack/training_args.html#CommandLineTrainingArgs
[1053]: #textattack.training_args.CommandLineTrainingArgs
[1054]: #textattack.training_args.TrainingArgs
[1055]: #textattack.training_args.CommandLineTrainingArgs.output_dir
[1056]: ../_modules/textattack/training_args.html#TrainingArgs
[1057]: #textattack.training_args.TrainingArgs
[1058]: #textattack.training_args.TrainingArgs.num_warmup_steps
[1059]: #textattack.training_args.TrainingArgs.num_train_adv_examples
[1060]: ../api/attacker.html#textattack.AttackArgs
[1061]: #textattack.training_args.TrainingArgs.load_best_model_at_end
[1062]: #textattack.training_args.TrainingArgs.alpha
[1063]: #textattack.training_args.TrainingArgs.attack_epoch_interval
[1064]: #textattack.training_args.TrainingArgs.attack_num_workers_per_device
[1065]: #textattack.training_args.TrainingArgs.checkpoint_interval_epochs
[1066]: #textattack.training_args.TrainingArgs.checkpoint_interval_steps
[1067]: #textattack.training_args.TrainingArgs.early_stopping_epochs
[1068]: #textattack.training_args.TrainingArgs.gradient_accumulation_steps
[1069]: #textattack.training_args.TrainingArgs.learning_rate
[1070]: #textattack.training_args.TrainingArgs.load_best_model_at_end
[1071]: #textattack.training_args.TrainingArgs.log_to_tb
[1072]: #textattack.training_args.TrainingArgs.log_to_wandb
[1073]: #textattack.training_args.TrainingArgs.logging_interval_step
[1074]: #textattack.training_args.TrainingArgs.num_clean_epochs
[1075]: #textattack.training_args.TrainingArgs.num_epochs
[1076]: #textattack.training_args.TrainingArgs.num_train_adv_examples
[1077]: #textattack.training_args.TrainingArgs.num_warmup_steps
[1078]: #textattack.training_args.TrainingArgs.output_dir
[1079]: #textattack.training_args.TrainingArgs.parallel
[1080]: #textattack.training_args.TrainingArgs.per_device_eval_batch_size
[1081]: #textattack.training_args.TrainingArgs.per_device_train_batch_size
[1082]: #textattack.training_args.TrainingArgs.query_budget_train
[1083]: #textattack.training_args.TrainingArgs.random_seed
[1084]: #textattack.training_args.TrainingArgs.save_last
[1085]: #textattack.training_args.TrainingArgs.tb_log_dir
[1086]: #textattack.training_args.TrainingArgs.wandb_project
[1087]: #textattack.training_args.TrainingArgs.weight_decay
[1088]: ../_modules/textattack/training_args.html#default_output_dir
[1089]: #textattack.training_args.default_output_dir
[1090]: ../api/search_methods.html
[1091]: textattack.attack_recipes.html
[1092]: https://www.sphinx-doc.org/
[1093]: https://github.com/readthedocs/sphinx_rtd_theme
[1094]: https://readthedocs.org
