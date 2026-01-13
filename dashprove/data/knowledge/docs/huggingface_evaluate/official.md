Evaluate documentation

Evaluate on the Hub

# Evaluate

🏡 View all docsAWS Trainium & InferentiaAccelerateArgillaAutoTrainBitsandbytesChat UIDataset
viewerDatasetsDeploying on AWSDiffusersDistilabelEvaluateGoogle CloudGoogle TPUsGradioHubHub Python
LibraryHuggingface.jsInference Endpoints (dedicated)Inference
ProvidersKernelsLeRobotLeaderboardsLightevalMicrosoft AzureOptimumPEFTSafetensorsSentence
TransformersTRLTasksText Embeddings InferenceText Generation
InferenceTokenizersTrackioTransformersTransformers.jssmolagentstimm
Search documentation
mainv0.4.6v0.3.0v0.2.3v0.1.2 EN
Get started
[🤗 Evaluate ][1]
Tutorials
[Installation ][2][A quick tour ][3]
How-to guides
[Choosing the right metric ][4][Adding new evaluations ][5][Using the evaluator ][6][Using the
evaluator with custom pipelines ][7][Creating an EvaluationSuite ][8]
Using 🤗 Evaluate with other ML frameworks
[Transformers ][9][Keras and Tensorflow ][10][scikit-learn ][11]
Conceptual guides
[Types of evaluations ][12][Considerations for model evaluation ][13]
Reference
[Main classes ][14][Loading methods ][15][Saving methods ][16][Hub methods ][17][Evaluator classes
][18][Visualization methods ][19][Logging methods ][20]
[Hugging Face's logo]
Join the Hugging Face community

and get access to the augmented documentation experience

Collaborate on models, datasets and Spaces
Faster examples with accelerated inference
Switch between documentation themes
[Sign Up][21]

to get started

# Evaluate on the Hub


[Evaluate on the Hub banner]

You can evaluate AI models on the Hub in multiple ways and this page will guide you through the
different options:

* **Community Leaderboards** bring together the best models for a given task or domain and make them
  accessible to everyone by ranking them.
* **Model Cards** provide a comprehensive overview of a model’s capabilities from the author’s
  perspective.
* **Libraries and Packages** give you the tools to evaluate your models on the Hub.

## Community Leaderboards

Community leaderboards show how a model performs on a given task or domain. For example, there are
leaderboards for question answering, reasoning, classification, vision, and audio. If you’re
tackling a new task, you can use a leaderboard to see how a model performs on it.

Here are some examples of community leaderboards:

──────┬─────┬───────────────────────────────────────────────────────────────────────────────────────
Leader│Model│Description                                                                            
board │Type │                                                                                       
──────┼─────┼───────────────────────────────────────────────────────────────────────────────────────
[MTEB]│Embed│The Massive Text Embedding Benchmark leaderboard compares 100+ text and image embedding
[22]  │ding │models across 1000+ languages. Refer to the publication of each selectable benchmark   
      │     │for details on metrics, languages, tasks, and task types. Anyone is welcome to add a   
      │     │model, add benchmarks, help improve zero-shot annotations, or propose other changes to 
      │     │the leaderboard.                                                                       
──────┼─────┼───────────────────────────────────────────────────────────────────────────────────────
[GAIA]│Agent│GAIA is a benchmark which aims at evaluating next-generation LLMs (LLMs with augmented 
[23]  │ic   │capabilities due to added tooling, efficient prompting, access to search, etc). (See   
      │     │[the paper][24] for more details.)                                                     
──────┼─────┼───────────────────────────────────────────────────────────────────────────────────────
[OpenV│Visio│The OpenVLM Leaderboard evaluates 272+ Vision-Language Models (including GPT-4v,       
LM    │n    │Gemini, QwenVLPlus, LLaVA) across 31 different multi-modal benchmarks using the        
Leader│Langu│VLMEvalKit framework. It focuses on open-source VLMs and publicly available API models.
board]│age  │                                                                                       
[25]  │Model│                                                                                       
      │s    │                                                                                       
──────┼─────┼───────────────────────────────────────────────────────────────────────────────────────
[Open │Audio│The Open ASR Leaderboard ranks and evaluates speech recognition models on the Hugging  
ASR   │     │Face Hub. Models are ranked based on their Average WER, from lowest to highest.        
Leader│     │                                                                                       
board]│     │                                                                                       
[26]  │     │                                                                                       
──────┼─────┼───────────────────────────────────────────────────────────────────────────────────────
[LLM-P│LLM  │The 🤗 LLM-Perf Leaderboard 🏋️ is a leaderboard at the intersection of quality and     
erf   │Perfo│performance. Its aim is to benchmark the performance (latency, throughput, memory &    
Leader│rmanc│energy) of Large Language Models (LLMs) with different hardware, backends and          
board]│e    │optimizations using Optimum-Benchmark.                                                 
[27]  │     │                                                                                       
──────┴─────┴───────────────────────────────────────────────────────────────────────────────────────

There are many more leaderboards on the Hub. Check out all the leaderboards via this [search][28] or
use this [dedicated Space][29] to find a leaderboard for your task.

## Model Cards

Model cards provide an overview of a model’s capabilities evaluated by the community or the model’s
author. They are a great way to understand a model’s capabilities and limitations.

[Qwen model card]

Unlike leaderboards, model card evaluation scores are often created by the author, rather than by
the community.

For information on reporting results, see details on [the Model Card Evaluation Results
metadata][30].

## Libraries and packages

There are a number of open-source libraries and packages that you can use to evaluate your models on
the Hub. These are useful if you want to evaluate a custom model or performance on a custom
evaluation task.

### LightEval

LightEval is a library for evaluating LLMs. It is designed to be comprehensive and customizable.
Visit the LightEval [repository][31] for more information.

For more recent evaluation approaches that are popular on the Hugging Face Hub that are currently
more actively maintained, check out [LightEval][32].

### 🤗 Evaluate

A library for easily evaluating machine learning models and datasets.

With a single line of code, you get access to dozens of evaluation methods for different domains
(NLP, Computer Vision, Reinforcement Learning, and more!). Be it on your local machine or in a
distributed training setup, you can evaluate your models in a consistent and reproducible way!

Visit the 🤗 Evaluate [organization][33] for a full list of available metrics. Each metric has a
dedicated Space with an interactive demo for how to use the metric, and a documentation card
detailing the metrics limitations and usage.

[
Tutorials

Learn the basics and become familiar with loading, computing, and saving with 🤗 Evaluate. Start
here if you are using 🤗 Evaluate for the first time!

][34] [
How-to guides

Practical guides to help you achieve a specific goal. Take a look at these guides to learn how to
use 🤗 Evaluate to solve real-world problems.

][35] [
Conceptual guides

High-level explanations for building a better understanding of important topics such as
considerations going into evaluating a model or dataset and the difference between metrics,
measurements, and comparisons.

][36] [
Reference

Technical descriptions of how 🤗 Evaluate classes and methods work.

][37]
[< > Update on GitHub][38]
[Installation→][39]
[Evaluate on the Hub][40] [Community Leaderboards][41] [Model Cards][42] [Libraries and
packages][43] [LightEval][44] [🤗 Evaluate][45]

[1]: /docs/evaluate/index
[2]: /docs/evaluate/installation
[3]: /docs/evaluate/a_quick_tour
[4]: /docs/evaluate/choosing_a_metric
[5]: /docs/evaluate/creating_and_sharing
[6]: /docs/evaluate/base_evaluator
[7]: /docs/evaluate/custom_evaluator
[8]: /docs/evaluate/evaluation_suite
[9]: /docs/evaluate/transformers_integrations
[10]: /docs/evaluate/keras_integrations
[11]: /docs/evaluate/sklearn_integrations
[12]: /docs/evaluate/types_of_evaluations
[13]: /docs/evaluate/considerations
[14]: /docs/evaluate/package_reference/main_classes
[15]: /docs/evaluate/package_reference/loading_methods
[16]: /docs/evaluate/package_reference/saving_methods
[17]: /docs/evaluate/package_reference/hub_methods
[18]: /docs/evaluate/package_reference/evaluator_classes
[19]: /docs/evaluate/package_reference/visualization_methods
[20]: /docs/evaluate/package_reference/logging_methods
[21]: /join
[22]: https://huggingface.co/spaces/mteb/leaderboard
[23]: https://huggingface.co/spaces/gaia-benchmark/leaderboard
[24]: https://arxiv.org/abs/2311.12983
[25]: https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
[26]: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
[27]: https://huggingface.co/spaces/llm-perf/leaderboard
[28]: https://huggingface.co/spaces?category=model-benchmarking
[29]: https://huggingface.co/spaces/OpenEvals/find-a-leaderboard
[30]: https://huggingface.co/docs/hub/en/model-cards#evaluation-results
[31]: https://github.com/huggingface/lighteval
[32]: https://github.com/huggingface/lighteval
[33]: https://huggingface.co/evaluate-metric
[34]: ./installation
[35]: ./choosing_a_metric
[36]: ./types_of_evaluations
[37]: ./package_reference/main_classes
[38]: https://github.com/huggingface/evaluate/blob/main/docs/source/index.mdx
[39]: /docs/evaluate/installation
[40]: #evaluate-on-the-hub
[41]: #community-leaderboards
[42]: #model-cards
[43]: #libraries-and-packages
[44]: #lighteval
[45]: #-evaluate
