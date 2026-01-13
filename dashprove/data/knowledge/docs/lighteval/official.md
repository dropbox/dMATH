
[[lighteval library logo]][1]

*Your go-to toolkit for lightning-fast, flexible LLM evaluation, from Hugging Face's Leaderboard and
Evals Team.*

[[Tests]][2] [[Quality]][3] [[Python versions]][4] [[License]][5] [[Version]][6]

[ [Documentation] ][7] [ [Open Benchmark Index] ][8]

**Lighteval** is your *all-in-one toolkit* for evaluating LLMs across multiple backends—whether your
model is being **served somewhere** or **already loaded in memory**. Dive deep into your model's
performance by saving and exploring *detailed, sample-by-sample results* to debug and see how your
models stack-up.

*Customization at your fingertips*: letting you either browse all our existing tasks and
[metrics][9] or effortlessly create your own [custom task][10] and [custom metric][11], tailored to
your needs.

## Available Tasks

Lighteval supports **1000+ evaluation tasks** across multiple domains and languages. Use [this
space][12] to find what you need, or, here's an overview of some *popular benchmarks*:

### 📚 **Knowledge**

* **General Knowledge**: MMLU, MMLU-Pro, MMMU, BIG-Bench
* **Question Answering**: TriviaQA, Natural Questions, SimpleQA, Humanity's Last Exam (HLE)
* **Specialized**: GPQA, AGIEval

### 🧮 **Math and Code**

* **Math Problems**: GSM8K, GSM-Plus, MATH, MATH500
* **Competition Math**: AIME24, AIME25
* **Multilingual Math**: MGSM (Grade School Math in 10+ languages)
* **Coding Benchmarks**: LCB (LiveCodeBench)

### 🎯 **Chat Model Evaluation**

* **Instruction Following**: IFEval, IFEval-fr
* **Reasoning**: MUSR, DROP (discrete reasoning)
* **Long Context**: RULER
* **Dialogue**: MT-Bench
* **Holistic Evaluation**: HELM, BIG-Bench

### 🌍 **Multilingual Evaluation**

* **Cross-lingual**: XTREME, Flores200 (200 languages), XCOPA, XQuAD
* **Language-specific**:
  
  * **Arabic**: ArabicMMLU
  * **Filipino**: FilBench
  * **French**: IFEval-fr, GPQA-fr, BAC-fr
  * **German**: German RAG Eval
  * **Serbian**: Serbian LLM Benchmark, OZ Eval
  * **Turkic**: TUMLU (9 Turkic languages)
  * **Chinese**: CMMLU, CEval, AGIEval
  * **Russian**: RUMMLU, Russian SQuAD
  * **Kyrgyz**: Kyrgyz LLM Benchmark
  * **And many more...**

### 🧠 **Core Language Understanding**

* **NLU**: GLUE, SuperGLUE, TriviaQA, Natural Questions
* **Commonsense**: HellaSwag, WinoGrande, ProtoQA
* **Natural Language Inference**: XNLI
* **Reading Comprehension**: SQuAD, XQuAD, MLQA, Belebele

## ⚡️ Installation

> **Note**: lighteval is currently *completely untested on Windows*, and we don't support it yet.
> (*Should be fully functional on Mac/Linux*)

pip install lighteval

Lighteval allows for *many extras* when installing, see [here][13] for a **complete list**.

If you want to push results to the **Hugging Face Hub**, add your access token as an environment
variable:

hf auth login

## 🚀 Quickstart

Lighteval offers the following entry points for model evaluation:

* `lighteval eval`: Evaluation models using [inspect-ai][14] as a backend (prefered).
* `lighteval accelerate`: Evaluate models on CPU or one or more GPUs using [🤗 Accelerate][15]
* `lighteval nanotron`: Evaluate models in distributed settings using [⚡️ Nanotron][16]
* `lighteval vllm`: Evaluate models on one or more GPUs using [🚀 VLLM][17]
* `lighteval sglang`: Evaluate models using [SGLang][18] as backend
* `lighteval endpoint`: Evaluate models using various endpoints as backend
  
  * `lighteval endpoint inference-endpoint`: Evaluate models using Hugging Face's [Inference
    Endpoints API][19]
  * `lighteval endpoint tgi`: Evaluate models using [🔗 Text Generation Inference][20] running
    locally
  * `lighteval endpoint litellm`: Evaluate models on any compatible API using [LiteLLM][21]
  * `lighteval endpoint inference-providers`: Evaluate models using [HuggingFace's inference
    providers][22] as backend

Did not find what you need ? You can always make your custom model API by following [this guide][23]

* `lighteval custom`: Evaluate custom models (can be anything)

Here's a **quick command** to evaluate using a remote inference service:

lighteval eval "hf-inference-providers/openai/gpt-oss-20b" gpqa:diamond

Or use the **Python API** to run a model *already loaded in memory*!

from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelCon
fig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
BENCHMARKS = "gsm8k"

evaluation_tracker = EvaluationTracker(output_dir="./results")
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE,
    max_samples=2
)

model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME, device_map="auto"
)
config = TransformersModelConfig(model_name=MODEL_NAME, batch_size=1)
model = TransformersModel.from_model(model, config)

pipeline = Pipeline(
    model=model,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    tasks=BENCHMARKS,
)

results = pipeline.evaluate()
pipeline.show_results()
results = pipeline.get_results()

## 🙏 Acknowledgements

Lighteval took inspiration from the following *amazing* frameworks: Eleuther's [AI Harness][24] and
Stanford's [HELM][25]. We are grateful to their teams for their **pioneering work** on LLM
evaluations.

We'd also like to offer our thanks to all the community members who have contributed to the library,
adding new features and reporting or fixing bugs.

## 🌟 Contributions Welcome 💙💚💛💜🧡

**Got ideas?** Found a bug? Want to add a [task][26] or [metric][27]? Contributions are *warmly
welcomed*!

If you're adding a **new feature**, please *open an issue first*.

If you open a PR, don't forget to **run the styling**!

pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files

## 📜 Citation

@misc{lighteval,
  author = {Habib, Nathan and Fourrier, Clémentine and Kydlíček, Hynek and Wolf, Thomas and Tunstall
, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.11.0},
  url = {https://github.com/huggingface/lighteval}
}

[1]: /huggingface/lighteval/blob/main/assets/lighteval-doc.svg
[2]: https://github.com/huggingface/lighteval/actions/workflows/tests.yaml?query=branch%3Amain
[3]: https://github.com/huggingface/lighteval/actions/workflows/quality.yaml?query=branch%3Amain
[4]: https://www.python.org/downloads/
[5]: https://github.com/huggingface/lighteval/blob/main/LICENSE
[6]: https://pypi.org/project/lighteval/
[7]: https://huggingface.co/docs/lighteval/main/en/index
[8]: https://huggingface.co/spaces/OpenEvals/open_benchmark_index
[9]: https://huggingface.co/docs/lighteval/metric-list
[10]: https://huggingface.co/docs/lighteval/adding-a-custom-task
[11]: https://huggingface.co/docs/lighteval/adding-a-new-metric
[12]: https://huggingface.co/spaces/OpenEvals/open_benchmark_index
[13]: https://huggingface.co/docs/lighteval/installation
[14]: https://inspect.aisi.org.uk/
[15]: https://github.com/huggingface/accelerate
[16]: https://github.com/huggingface/nanotron
[17]: https://github.com/vllm-project/vllm
[18]: https://github.com/sgl-project/sglang
[19]: https://huggingface.co/inference-endpoints/dedicated
[20]: https://huggingface.co/docs/text-generation-inference/en/index
[21]: https://www.litellm.ai/
[22]: https://huggingface.co/docs/inference-providers/en/index
[23]: https://huggingface.co/docs/lighteval/main/en/evaluating-a-custom-model
[24]: https://github.com/EleutherAI/lm-evaluation-harness
[25]: https://crfm.stanford.edu/helm/latest/
[26]: https://huggingface.co/docs/lighteval/adding-a-custom-task
[27]: https://huggingface.co/docs/lighteval/adding-a-new-metric
