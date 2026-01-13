# Language Model Evaluation Harness

[[DOI]][1]

## Latest News 📣

* [2025/12] **CLI refactored** with subcommands (`run`, `ls`, `validate`) and YAML config file
  support via `--config`. See the [CLI Reference][2] and [Configuration Guide][3].
* [2025/12] **Lighter install**: Base package no longer includes `transformers`/`torch`. Install
  model backends separately: `pip install lm_eval[hf]`, `lm_eval[vllm]`, etc.
* [2025/07] Added `think_end_token` arg to `hf` (token/str), `vllm` and `sglang` (str) for stripping
  CoT reasoning traces from models that support it.
* [2025/03] Added support for steering HF models!
* [2025/02] Added [SGLang][4] support!
* [2024/09] We are prototyping allowing users of LM Evaluation Harness to create and evaluate on
  text+image multimodal input, text output tasks, and have just added the `hf-multimodal` and
  `vllm-vlm` model types and `mmmu` task as a prototype feature. We welcome users to try out this
  in-progress feature and stress-test it for themselves, and suggest they check out
  [`lmms-eval`][5], a wonderful project originally forking off of the lm-evaluation-harness, for a
  broader range of multimodal tasks, models, and features.
* [2024/07] [API model][6] support has been updated and refactored, introducing support for batched
  and async requests, and making it significantly easier to customize and use for your own purposes.
  **To run Llama 405B, we recommend using VLLM's OpenAI-compliant API to host the model, and use the
  `local-completions` model type to evaluate the model.**
* [2024/07] New Open LLM Leaderboard tasks have been added ! You can find them under the
  [leaderboard][7] task group.

## Announcement

**A new v0.4.0 release of lm-evaluation-harness is available** !

New updates and features include:

* **New Open LLM Leaderboard tasks have been added ! You can find them under the [leaderboard][8]
  task group.**
* Internal refactoring
* Config-based task creation and configuration
* Easier import and sharing of externally-defined task config YAMLs
* Support for Jinja2 prompt design, easy modification of prompts + prompt imports from Promptsource
* More advanced configuration options, including output post-processing, answer extraction, and
  multiple LM generations per document, configurable fewshot settings, and more
* Speedups and new modeling libraries supported, including: faster data-parallel HF model usage,
  vLLM support, MPS support with HuggingFace, and more
* Logging and usability changes
* New tasks including CoT BIG-Bench-Hard, Belebele, user-defined task groupings, and more

Please see our updated documentation pages in `docs/` for more details.

Development will be continuing on the `main` branch, and we encourage you to give us feedback on
what features are desired and how to improve the library further, or ask questions, either in issues
or PRs on GitHub, or in the [EleutherAI discord][9]!

## Overview

This project provides a unified framework to test generative language models on a large number of
different evaluation tasks.

**Features:**

* Over 60 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented.
* Support for models loaded via [transformers][10] (including quantization via [GPTQModel][11] and
  [AutoGPTQ][12]), [GPT-NeoX][13], and [Megatron-DeepSpeed][14], with a flexible
  tokenization-agnostic interface.
* Support for fast and memory-efficient inference with [vLLM][15].
* Support for commercial APIs including [OpenAI][16], and [TextSynth][17].
* Support for evaluation on adapters (e.g. LoRA) supported in [HuggingFace's PEFT library][18].
* Support for local models and benchmarks.
* Evaluation with publicly available prompts ensures reproducibility and comparability between
  papers.
* Easy support for custom prompts and evaluation metrics.

The Language Model Evaluation Harness is the backend for 🤗 Hugging Face's popular [Open LLM
Leaderboard][19], has been used in [hundreds of papers][20], and is used internally by dozens of
organizations including NVIDIA, Cohere, BigScience, BigCode, Nous Research, and Mosaic ML.

## Install

To install the `lm-eval` package from the github repository, run:

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

### Installing Model Backends

The base installation provides the core evaluation framework. **Model backends must be installed
separately** using optional extras:

For HuggingFace transformers models:

pip install "lm_eval[hf]"

For vLLM inference:

pip install "lm_eval[vllm]"

For API-based models (OpenAI, Anthropic, etc.):

pip install "lm_eval[api]"

Multiple backends can be installed together:

pip install "lm_eval[hf,vllm,api]"

A detailed table of all optional extras is available at the end of this document.

## Basic Usage

### Documentation

────────────────────────┬────────────────────────────────────────
Guide                   │Description                             
────────────────────────┼────────────────────────────────────────
[CLI Reference][21]     │Command-line arguments and subcommands  
────────────────────────┼────────────────────────────────────────
[Configuration          │YAML config file format and examples    
Guide][22]              │                                        
────────────────────────┼────────────────────────────────────────
[Python API][23]        │Programmatic usage with                 
                        │`simple_evaluate()`                     
────────────────────────┼────────────────────────────────────────
[Task Guide][24]        │Available tasks and task configuration  
────────────────────────┴────────────────────────────────────────

Use `lm-eval -h` to see available options, or `lm-eval run -h` for evaluation options.

List available tasks with:

lm-eval ls tasks

### Hugging Face `transformers`

Important

To use the HuggingFace backend, first install: `pip install "lm_eval[hf]"`

To evaluate a model hosted on the [HuggingFace Hub][25] (e.g. GPT-J-6B) on `hellaswag` you can use
the following command (this assumes you are using a CUDA-compatible GPU):

lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most
notably, this supports the common practice of using the `revisions` feature on the Hub to store
partially trained checkpoints, or to specify the datatype for running a model:

lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8

Models that are loaded via both `transformers.AutoModelForCausalLM` (autoregressive, decoder-only
GPT style models) and `transformers.AutoModelForSeq2SeqLM` (such as encoder-decoder models like T5)
in Huggingface are supported.

Batch size selection can be automated by setting the `--batch_size` flag to `auto`. This will
perform automatic detection of the largest batch size that will fit on your device. On tasks where
there is a large difference between the longest and shortest example, it can be helpful to
periodically recompute the largest batch size, to gain a further speedup. To do this, append `:N` to
above flag to automatically recompute the largest batch size `N` times. For example, to recompute
the batch size 4 times, the command would be:

lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size auto:4

Note

Just like you can provide a local path to `transformers.AutoModel`, you can also provide a local
path to `lm_eval` via `--model_args pretrained=/path/to/model`

#### Evaluating GGUF Models

`lm-eval` supports evaluating models in GGUF format using the Hugging Face (`hf`) backend. This
allows you to use quantized models compatible with `transformers`, `AutoModel`, and llama.cpp
conversions.

To evaluate a GGUF model, pass the path to the directory containing the model weights, the
`gguf_file`, and optionally a separate `tokenizer` path using the `--model_args` flag.

**🚨 Important Note:**
If no separate tokenizer is provided, Hugging Face will attempt to reconstruct the tokenizer from
the GGUF file — this can take **hours** or even hang indefinitely. Passing a separate tokenizer
avoids this issue and can reduce tokenizer loading time from hours to seconds.

**✅ Recommended usage:**

lm_eval --model hf \
    --model_args pretrained=/path/to/gguf_folder,gguf_file=model-name.gguf,tokenizer=/path/to/tokeni
zer \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8

Tip

Ensure the tokenizer path points to a valid Hugging Face tokenizer directory (e.g., containing
tokenizer_config.json, vocab.json, etc.).

#### Multi-GPU Evaluation with Hugging Face `accelerate`

We support three main ways of using Hugging Face's [accelerate 🚀][26] library for multi-GPU
evaluation.

To perform *data-parallel evaluation* (where each GPU loads a **separate full copy** of the model),
we leverage the `accelerate` launcher as follows:

accelerate launch -m lm_eval --model hf \
    --tasks lambada_openai,arc_easy \
    --batch_size 16

(or via `accelerate launch --no-python lm_eval`).

For cases where your model can fit on a single GPU, this allows you to evaluate on K GPUs K times
faster than on one.

**WARNING**: This setup does not work with FSDP model sharding, so in `accelerate config` FSDP must
be disabled, or the NO_SHARD FSDP option must be used.

The second way of using `accelerate` for multi-GPU evaluation is when your model is *too large to
fit on a single GPU.*

In this setting, run the library *outside the `accelerate` launcher*, but passing `parallelize=True`
to `--model_args` as follows:

lm_eval --model hf \
    --tasks lambada_openai,arc_easy \
    --model_args parallelize=True \
    --batch_size 16

This means that your model's weights will be split across all available GPUs.

For more advanced users or even larger models, we allow for the following arguments when
`parallelize=True` as well:

* `device_map_option`: How to split model weights across available GPUs. defaults to "auto".
* `max_memory_per_gpu`: the max GPU memory to use per GPU in loading the model.
* `max_cpu_memory`: the max amount of CPU memory to use when offloading the model weights to RAM.
* `offload_folder`: a folder where model weights will be offloaded to disk if needed.

The third option is to use both at the same time. This will allow you to take advantage of both data
parallelism and model sharding, and is especially useful for models that are too large to fit on a
single GPU.

accelerate launch --multi_gpu --num_processes {nb_of_copies_of_your_model} \
    -m lm_eval --model hf \
    --tasks lambada_openai,arc_easy \
    --model_args parallelize=True \
    --batch_size 16

To learn more about model parallelism and how to use it with the `accelerate` library, see the
[accelerate documentation][27]

**Warning: We do not natively support multi-node evaluation using the `hf` model type! Please
reference [our GPT-NeoX library integration][28] for an example of code in which a custom
multi-machine evaluation script is written.**

**Note: we do not currently support multi-node evaluations natively, and advise using either an
externally hosted server to run inference requests against, or creating a custom integration with
your distributed framework [as is done for the GPT-NeoX library][29].**

### Steered Hugging Face `transformers` models

To evaluate a Hugging Face `transformers` model with steering vectors applied, specify the model
type as `steered` and provide the path to either a PyTorch file containing pre-defined steering
vectors, or a CSV file that specifies how to derive steering vectors from pretrained `sparsify` or
`sae_lens` models (you will need to install the corresponding optional dependency for this method).

Specify pre-defined steering vectors:

import torch

steer_config = {
    "layers.3": {
        "steering_vector": torch.randn(1, 768),
        "bias": torch.randn(1, 768),
        "steering_coefficient": 1,
        "action": "add"
    },
}
torch.save(steer_config, "steer_config.pt")

Specify derived steering vectors:

import pandas as pd

pd.DataFrame({
    "loader": ["sparsify"],
    "action": ["add"],
    "sparse_model": ["EleutherAI/sae-pythia-70m-32k"],
    "hookpoint": ["layers.3"],
    "feature_index": [30],
    "steering_coefficient": [10.0],
}).to_csv("steer_config.csv", index=False)

Run the evaluation harness with steering vectors applied:

lm_eval --model steered \
    --model_args pretrained=EleutherAI/pythia-160m,steer_path=steer_config.pt \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8

### NVIDIA `nemo` models

[NVIDIA NeMo Framework][30] is a generative AI framework built for researchers and pytorch
developers working on language models.

To evaluate a `nemo` model, start by installing NeMo following [the documentation][31]. We highly
recommended to use the NVIDIA PyTorch or NeMo container, especially if having issues installing Apex
or any other dependencies (see [latest released containers][32]). Please also install the lm
evaluation harness library following the instructions in [the Install section][33].

NeMo models can be obtained through [NVIDIA NGC Catalog][34] or in [NVIDIA's Hugging Face page][35].
In [NVIDIA NeMo Framework][36] there are conversion scripts to convert the `hf` checkpoints of
popular models like llama, falcon, mixtral or mpt to `nemo`.

Run a `nemo` model on one GPU:

lm_eval --model nemo_lm \
    --model_args path=<path_to_nemo_model> \
    --tasks hellaswag \
    --batch_size 32

It is recommended to unpack the `nemo` model to avoid the unpacking inside the docker container - it
may overflow disk space. For that you can run:

mkdir MY_MODEL
tar -xvf MY_MODEL.nemo -c MY_MODEL

#### Multi-GPU evaluation with NVIDIA `nemo` models

By default, only one GPU is used. But we do support either data replication or tensor/pipeline
parallelism during evaluation, on one node.

1. To enable data replication, set the `model_args` of `devices` to the number of data replicas to
   run. For example, the command to run 8 data replicas over 8 GPUs is:
torchrun --nproc-per-node=8 --no-python lm_eval \
    --model nemo_lm \
    --model_args path=<path_to_nemo_model>,devices=8 \
    --tasks hellaswag \
    --batch_size 32

1. To enable tensor and/or pipeline parallelism, set the `model_args` of
   `tensor_model_parallel_size` and/or `pipeline_model_parallel_size`. In addition, you also have to
   set up `devices` to be equal to the product of `tensor_model_parallel_size` and/or
   `pipeline_model_parallel_size`. For example, the command to use one node of 4 GPUs with tensor
   parallelism of 2 and pipeline parallelism of 2 is:
torchrun --nproc-per-node=4 --no-python lm_eval \
    --model nemo_lm \
    --model_args path=<path_to_nemo_model>,devices=4,tensor_model_parallel_size=2,pipeline_model_par
allel_size=2 \
    --tasks hellaswag \
    --batch_size 32

Note that it is recommended to substitute the `python` command by `torchrun --nproc-per-node=<number
of devices> --no-python` to facilitate loading the model into the GPUs. This is especially important
for large checkpoints loaded into multiple GPUs.

Not supported yet: multi-node evaluation and combinations of data replication with tensor or
pipeline parallelism.

#### Multi-GPU evaluation with OpenVINO models

Pipeline parallelism during evaluation is supported with OpenVINO models

To enable pipeline parallelism, set the `model_args` of `pipeline_parallel`. In addition, you also
have to set up `device` to value `HETERO:<GPU index1>,<GPU index2>` for example `HETERO:GPU.1,GPU.0`
For example, the command to use pipeline parallelism of 2 is:

lm_eval --model openvino \
    --tasks wikitext \
    --model_args pretrained=<path_to_ov_model>,pipeline_parallel=True \
    --device HETERO:GPU.1,GPU.0

### Tensor + Data Parallel and Optimized Inference with `vLLM`

We also support vLLM for faster inference on [supported model types][37], especially faster when
splitting a model across multiple GPUs. For single-GPU or multi-GPU — tensor parallel, data
parallel, or a combination of both — inference, for example:

lm_eval --model vllm \
    --model_args pretrained={model_name},tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory
_utilization=0.8,data_parallel_size={model_replicas} \
    --tasks lambada_openai \
    --batch_size auto

To use vllm, do `pip install "lm_eval[vllm]"`. For a full list of supported vLLM configurations,
please reference our [vLLM integration][38] and the vLLM documentation.

vLLM occasionally differs in output from Huggingface. We treat Huggingface as the reference
implementation and provide a [script][39] for checking the validity of vllm results against HF.

Tip

For fastest performance, we recommend using `--batch_size auto` for vLLM whenever possible, to
leverage its continuous batching functionality!

Tip

Passing `max_model_len=4096` or some other reasonable default to vLLM through model args may cause
speedups or prevent out-of-memory errors when trying to use auto batch size, such as for
Mistral-7B-v0.1 which defaults to a maximum length of 32k.

### Tensor + Data Parallel and Fast Offline Batching Inference with `SGLang`

We support SGLang for efficient offline batch inference. Its **[Fast Backend Runtime][40]** delivers
high performance through optimized memory management and parallel processing techniques. Key
features include tensor parallelism, continuous batching, and support for various quantization
methods (FP8/INT4/AWQ/GPTQ).

To use SGLang as the evaluation backend, please **install it in advance** via SGLang documents
[here][41].

Tip

Due to the installing method of [`Flashinfer`][42]-- a fast attention kernel library, we don't
include the dependencies of `SGLang` within [pyproject.toml][43]. Note that the `Flashinfer` also
has some requirements on `torch` version.

SGLang's server arguments are slightly different from other backends, see [here][44] for more
information. We provide an example of the usage here:

lm_eval --model sglang \
    --model_args pretrained={model_name},dp_size={data_parallel_size},tp_size={tensor_parallel_size}
,dtype=auto \
    --tasks gsm8k_cot \
    --batch_size auto

Tip

When encountering out-of-memory (OOM) errors (especially for multiple-choice tasks), try these
solutions:

1. Use a manual `batch_size`, rather than `auto`.
2. Lower KV cache pool memory usage by adjusting `mem_fraction_static` - Add to your model arguments
   for example `--model_args pretrained=...,mem_fraction_static=0.7`.
3. Increase tensor parallel size `tp_size` (if using multiple GPUs).

### Model APIs and Inference Servers

Important

To use API-based models, first install: `pip install "lm_eval[api]"`

Our library also supports the evaluation of models served via several commercial APIs, and we hope
to implement support for the most commonly used performant local/self-hosted inference servers.

To call a hosted model, use:

export OPENAI_API_KEY=YOUR_KEY_HERE
lm_eval --model openai-completions \
    --model_args model=davinci-002 \
    --tasks lambada_openai,hellaswag

We also support using your own local inference server with servers that mirror the OpenAI
Completions and ChatCompletions APIs.

lm_eval --model local-completions --tasks gsm8k --model_args model=facebook/opt-125m,base_url=http:/
/{yourip}:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=16

Note that for externally hosted models, configs such as `--device` which relate to where to place a
local model should not be used and do not function. Just like you can use `--model_args` to pass
arbitrary arguments to the model constructor for local models, you can use it to pass arbitrary
arguments to the model API for hosted models. See the documentation of the hosting service for
information on what arguments they support.

─────────────┬──────────┬──────────────┬─────────────────────────────────────┬──────────────────────
API or       │Implemente│`--model      │Models supported:                    │Request Types:        
Inference    │d?        │<xxx>` name   │                                     │                      
Server       │          │              │                                     │                      
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
OpenAI       │✔️        │`openai-comple│All OpenAI Completions API models    │`generate_until`,     
Completions  │          │tions`,       │                                     │`loglikelihood`,      
             │          │`local-complet│                                     │`loglikelihood_rolling
             │          │ions`         │                                     │`                     
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
OpenAI       │✔️        │`openai-chat-c│[All ChatCompletions API models][45] │`generate_until` (no  
ChatCompletio│          │ompletions`,  │                                     │logprobs)             
ns           │          │`local-chat-co│                                     │                      
             │          │mpletions`    │                                     │                      
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
Anthropic    │✔️        │`anthropic`   │[Supported Anthropic Engines][46]    │`generate_until` (no  
             │          │              │                                     │logprobs)             
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
Anthropic    │✔️        │`anthropic-cha│[Supported Anthropic Engines][47]    │`generate_until` (no  
Chat         │          │t`,           │                                     │logprobs)             
             │          │`anthropic-cha│                                     │                      
             │          │t-completions`│                                     │                      
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
Textsynth    │✔️        │`textsynth`   │[All supported engines][48]          │`generate_until`,     
             │          │              │                                     │`loglikelihood`,      
             │          │              │                                     │`loglikelihood_rolling
             │          │              │                                     │`                     
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
Cohere       │[⌛ -     │N/A           │[All `cohere.generate()` engines][50]│`generate_until`,     
             │blocked on│              │                                     │`loglikelihood`,      
             │Cohere API│              │                                     │`loglikelihood_rolling
             │bug][49]  │              │                                     │`                     
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
[Llama.cpp][5│✔️        │`gguf`, `ggml`│[All models supported by             │`generate_until`,     
1] (via      │          │              │llama.cpp][53]                       │`loglikelihood`,      
[llama-cpp-py│          │              │                                     │(perplexity evaluation
thon][52])   │          │              │                                     │not yet implemented)  
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
vLLM         │✔️        │`vllm`        │[Most HF Causal Language Models][54] │`generate_until`,     
             │          │              │                                     │`loglikelihood`,      
             │          │              │                                     │`loglikelihood_rolling
             │          │              │                                     │`                     
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
Mamba        │✔️        │`mamba_ssm`   │[Mamba architecture Language Models  │`generate_until`,     
             │          │              │via the `mamba_ssm` package][55]     │`loglikelihood`,      
             │          │              │                                     │`loglikelihood_rolling
             │          │              │                                     │`                     
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
Huggingface  │✔️        │`openvino`    │Any decoder-only AutoModelForCausalLM│`generate_until`,     
Optimum      │          │              │converted with Huggingface Optimum   │`loglikelihood`,      
(Causal LMs) │          │              │into OpenVINO™ Intermediate          │`loglikelihood_rolling
             │          │              │Representation (IR) format           │`                     
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
Huggingface  │✔️        │`ipex`        │Any decoder-only AutoModelForCausalLM│`generate_until`,     
Optimum-intel│          │              │                                     │`loglikelihood`,      
IPEX (Causal │          │              │                                     │`loglikelihood_rolling
LMs)         │          │              │                                     │`                     
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
Neuron via   │✔️        │`neuronx`     │Any decoder-only AutoModelForCausalLM│`generate_until`,     
AWS Inf2     │          │              │supported to run on [huggingface-ami │`loglikelihood`,      
(Causal LMs) │          │              │image for inferentia2][56]           │`loglikelihood_rolling
             │          │              │                                     │`                     
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
NVIDIA NeMo  │✔️        │`nemo_lm`     │[All supported models][57]           │`generate_until`,     
             │          │              │                                     │`loglikelihood`,      
             │          │              │                                     │`loglikelihood_rolling
             │          │              │                                     │`                     
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
Watsonx.ai   │✔️        │`watsonx_llm` │[Supported Watsonx.ai Engines][58]   │`generate_until`      
             │          │              │                                     │`loglikelihood`       
─────────────┼──────────┼──────────────┼─────────────────────────────────────┼──────────────────────
[Your local  │✔️        │`local-complet│Support for OpenAI API-compatible    │`generate_until`,     
inference    │          │ions` or      │servers, with easy customization for │`loglikelihood`,      
server!][59] │          │`local-chat-co│other APIs.                          │`loglikelihood_rolling
             │          │mpletions`    │                                     │`                     
─────────────┴──────────┴──────────────┴─────────────────────────────────────┴──────────────────────

Models which do not supply logits or logprobs can be used with tasks of type `generate_until` only,
while local models, or APIs that supply logprobs/logits of their prompts, can be run on all task
types: `generate_until`, `loglikelihood`, `loglikelihood_rolling`, and `multiple_choice`.

For more information on the different task `output_types` and model request types, see [our
documentation][60].

Note

For best performance with closed chat model APIs such as Anthropic Claude 3 and GPT-4, we recommend
carefully looking at a few sample outputs using `--limit 10` first to confirm answer extraction and
scoring on generative tasks is performing as expected. providing `system="<some system prompt
here>"` within `--model_args` for anthropic-chat-completions, to instruct the model what format to
respond in, may be useful.

### Other Frameworks

A number of other libraries contain scripts for calling the eval harness through their library.
These include [GPT-NeoX][61], [Megatron-DeepSpeed][62], and [mesh-transformer-jax][63].

To create your own custom integration you can follow instructions from [this tutorial][64].

### Additional Features

Note

For tasks unsuitable for direct evaluation — either due risks associated with executing untrusted
code or complexities in the evaluation process — the `--predict_only` flag is available to obtain
decoded generations for post-hoc evaluation.

If you have a Metal compatible Mac, you can run the eval harness using the MPS back-end by replacing
`--device cuda:0` with `--device mps` (requires PyTorch version 2.1 or higher). **Note that the
PyTorch MPS backend is still in early stages of development, so correctness issues or unsupported
operations may exist. If you observe oddities in model performance on the MPS back-end, we recommend
first checking that a forward pass of your model on `--device cpu` and `--device mps` match.**

Note

You can inspect what the LM inputs look like by running the following command:

python write_out.py \
    --tasks <task1,task2,...> \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder

This will write out one text file for each task.

To verify the data integrity of the tasks you're performing in addition to running the tasks
themselves, you can use the `--check_integrity` flag:

lm_eval --model openai \
    --model_args engine=davinci-002 \
    --tasks lambada_openai,hellaswag \
    --check_integrity

## Advanced Usage Tips

For models loaded with the HuggingFace `transformers` library, any arguments provided via
`--model_args` get passed to the relevant constructor directly. This means that anything you can do
with `AutoModel` can be done with our library. For example, you can pass a local path via
`pretrained=` or use models finetuned with [PEFT][65] by taking the call you would run to evaluate
the base model and add `,peft=PATH` to the `model_args` argument:

lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt
4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0

Models provided as delta weights can be easily loaded using the Hugging Face transformers library.
Within --model_args, set the delta argument to specify the delta weights, and use the pretrained
argument to designate the relative base model to which they will be applied:

lm_eval --model hf \
    --model_args pretrained=Ejafa/llama_7B,delta=lmsys/vicuna-7b-delta-v1.1 \
    --tasks hellaswag

GPTQ quantized models can be loaded using [GPTQModel][66] (faster) or [AutoGPTQ][67]

GPTQModel: add `,gptqmodel=True` to `model_args`

lm_eval --model hf \
    --model_args pretrained=model-name-or-path,gptqmodel=True \
    --tasks hellaswag

AutoGPTQ: add `,autogptq=True` to `model_args`:

lm_eval --model hf \
    --model_args pretrained=model-name-or-path,autogptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag

We support wildcards in task names, for example you can run all of the machine-translated lambada
tasks via `--task lambada_openai_mt_*`.

## Saving & Caching Results

To save evaluation results provide an `--output_path`. We also support logging model responses with
the `--log_samples` flag for post-hoc analysis.

Tip

Use `--use_cache <DIR>` to cache evaluation results and skip previously evaluated samples when
resuming runs of the same (model, task) pairs. Note that caching is rank-dependent, so restart with
the same GPU count if interrupted. You can also use --cache_requests to save dataset preprocessing
steps for faster evaluation resumption.

To push results and samples to the Hugging Face Hub, first ensure an access token with write access
is set in the `HF_TOKEN` environment variable. Then, use the `--hf_hub_log_args` flag to specify the
organization, repository name, repository visibility, and whether to push results and samples to the
Hub - [example dataset on the HF Hub][68]. For instance:

lm_eval --model hf \
    --model_args pretrained=model-name-or-path,autogptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag \
    --log_samples \
    --output_path results \
    --hf_hub_log_args hub_results_org=EleutherAI,hub_repo_name=lm-eval-results,push_results_to_hub=T
rue,push_samples_to_hub=True,public_repo=False \

This allows you to easily download the results and samples from the Hub, using:

from datasets import load_dataset

load_dataset("EleutherAI/lm-eval-results-private", "hellaswag", "latest")

For a full list of supported arguments, check out the [interface][69] guide in our documentation!

## Visualizing Results

You can seamlessly visualize and analyze the results of your evaluation harness runs using both
Weights & Biases (W&B) and Zeno.

### Zeno

You can use [Zeno][70] to visualize the results of your eval harness runs.

First, head to [hub.zenoml.com][71] to create an account and get an API key [on your account
page][72]. Add this key as an environment variable:

export ZENO_API_KEY=[your api key]

You'll also need to install the `lm_eval[zeno]` package extra.

To visualize the results, run the eval harness with the `log_samples` and `output_path` flags. We
expect `output_path` to contain multiple folders that represent individual model names. You can thus
run your evaluation on any number of tasks and models and upload all of the results as projects on
Zeno.

lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --output_path output/gpt-j-6B

Then, you can upload the resulting data using the `zeno_visualize` script:

python scripts/zeno_visualize.py \
    --data_path output \
    --project_name "Eleuther Project"

This will use all subfolders in `data_path` as different models and upload all tasks within these
model folders to Zeno. If you run the eval harness on multiple tasks, the `project_name` will be
used as a prefix and one project will be created per task.

You can find an example of this workflow in [examples/visualize-zeno.ipynb][73].

### Weights and Biases

With the [Weights and Biases][74] integration, you can now spend more time extracting deeper
insights into your evaluation results. The integration is designed to streamline the process of
logging and visualizing experiment results using the Weights & Biases (W&B) platform.

The integration provide functionalities

* to automatically log the evaluation results,
* log the samples as W&B Tables for easy visualization,
* log the `results.json` file as an artifact for version control,
* log the `<task_name>_eval_samples.json` file if the samples are logged,
* generate a comprehensive report for analysis and visualization with all the important metric,
* log task and cli specific configs,
* and more out of the box like the command used to run the evaluation, GPU/CPU counts, timestamp,
  etc.

First you'll need to install the lm_eval[wandb] package extra. Do `pip install lm_eval[wandb]`.

Authenticate your machine with an your unique W&B token. Visit [https://wandb.ai/authorize][75] to
get one. Do `wandb login` in your command line terminal.

Run eval harness as usual with a `wandb_args` flag. Use this flag to provide arguments for
initializing a wandb run ([wandb.init][76]) as comma separated string arguments.

lm_eval \
    --model hf \
    --model_args pretrained=microsoft/phi-2,trust_remote_code=True \
    --tasks hellaswag,mmlu_abstract_algebra \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output/phi-2 \
    --limit 10 \
    --wandb_args project=lm-eval-harness-integration \
    --log_samples

In the stdout, you will find the link to the W&B run page as well as link to the generated report.
You can find an example of this workflow in [examples/visualize-wandb.ipynb][77], and an example of
how to integrate it beyond the CLI.

## Contributing

Check out our [open issues][78] and feel free to submit pull requests!

For more information on the library and how everything fits together, see our [documentation
pages][79].

To get started with development, first clone the repository and install the dev dependencies:

git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev,hf]"

### Implementing new tasks

To implement a new task in the eval harness, see [this guide][80].

In general, we follow this priority list for addressing concerns about prompting and other eval
details:

1. If there is widespread agreement among people who train LLMs, use the agreed upon procedure.
2. If there is a clear and unambiguous official implementation, use that procedure.
3. If there is widespread agreement among people who evaluate LLMs, use the agreed upon procedure.
4. If there are multiple common implementations but not universal or widespread agreement, use our
   preferred option among the common implementations. As before, prioritize choosing from among the
   implementations found in LLM training papers.

These are guidelines and not rules, and can be overruled in special circumstances.

We try to prioritize agreement with the procedures used by other groups to decrease the harm when
people inevitably compare runs across different papers despite our discouragement of the practice.
Historically, we also prioritized the implementation from [Language Models are Few Shot
Learners][81] as our original goal was specifically to compare results with that paper.

### Support

The best way to get support is to open an issue on this repo or join the [EleutherAI Discord
server][82]. The `#lm-thunderdome` channel is dedicated to developing this project and the
`#release-discussion` channel is for receiving support for our releases. If you've used the library
and have had a positive (or negative) experience, we'd love to hear from you!

## Optional Extras

Extras dependencies can be installed via `pip install -e ".[NAME]"`

### Model Backends

These extras install dependencies required to run specific model backends:

──────────────┬────────────────────────────────────────────────────────────────
NAME          │Description                                                     
──────────────┼────────────────────────────────────────────────────────────────
hf            │HuggingFace Transformers (torch, transformers, accelerate, peft)
──────────────┼────────────────────────────────────────────────────────────────
vllm          │vLLM fast inference                                             
──────────────┼────────────────────────────────────────────────────────────────
api           │API models (OpenAI, Anthropic, local servers)                   
──────────────┼────────────────────────────────────────────────────────────────
gptq          │AutoGPTQ quantized models                                       
──────────────┼────────────────────────────────────────────────────────────────
gptqmodel     │GPTQModel quantized models                                      
──────────────┼────────────────────────────────────────────────────────────────
ibm_watsonx_ai│IBM watsonx.ai models                                           
──────────────┼────────────────────────────────────────────────────────────────
ipex          │Intel IPEX backend                                              
──────────────┼────────────────────────────────────────────────────────────────
optimum       │Intel OpenVINO models                                           
──────────────┼────────────────────────────────────────────────────────────────
neuronx       │AWS Inferentia2 instances                                       
──────────────┼────────────────────────────────────────────────────────────────
sparsify      │Sparsify model steering                                         
──────────────┼────────────────────────────────────────────────────────────────
sae_lens      │SAELens model steering                                          
──────────────┴────────────────────────────────────────────────────────────────

### Task Dependencies

These extras install dependencies required for specific evaluation tasks:

────────────────────┬──────────────────────────────
NAME                │Description                   
────────────────────┼──────────────────────────────
tasks               │All task-specific dependencies
────────────────────┼──────────────────────────────
acpbench            │ACP Bench tasks               
────────────────────┼──────────────────────────────
audiolm_qwen        │Qwen2 audio models            
────────────────────┼──────────────────────────────
ifeval              │IFEval task                   
────────────────────┼──────────────────────────────
japanese_leaderboard│Japanese LLM tasks            
────────────────────┼──────────────────────────────
longbench           │LongBench tasks               
────────────────────┼──────────────────────────────
math                │Math answer checking          
────────────────────┼──────────────────────────────
multilingual        │Multilingual tokenizers       
────────────────────┼──────────────────────────────
ruler               │RULER tasks                   
────────────────────┴──────────────────────────────

### Development & Utilities

─────────────┬─────────────────────────
NAME         │Description              
─────────────┼─────────────────────────
dev          │Linting & contributions  
─────────────┼─────────────────────────
hf_transfer  │Speed up HF downloads    
─────────────┼─────────────────────────
sentencepiece│Sentencepiece tokenizer  
─────────────┼─────────────────────────
unitxt       │Unitxt tasks             
─────────────┼─────────────────────────
wandb        │Weights & Biases logging 
─────────────┼─────────────────────────
zeno         │Zeno result visualization
─────────────┴─────────────────────────

## Cite as

`@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid a
nd DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain a
nd Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reyn
olds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite
, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
`

[1]: https://doi.org/10.5281/zenodo.10256836
[2]: /EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
[3]: /EleutherAI/lm-evaluation-harness/blob/main/docs/config_files.md
[4]: https://docs.sglang.ai/
[5]: https://github.com/EvolvingLMMs-Lab/lmms-eval
[6]: /EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md
[7]: /EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/README.md
[8]: /EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/README.md
[9]: https://discord.gg/eleutherai
[10]: https://github.com/huggingface/transformers/
[11]: https://github.com/ModelCloud/GPTQModel
[12]: https://github.com/PanQiWei/AutoGPTQ
[13]: https://github.com/EleutherAI/gpt-neox
[14]: https://github.com/microsoft/Megatron-DeepSpeed/
[15]: https://github.com/vllm-project/vllm
[16]: https://openai.com
[17]: https://textsynth.com/
[18]: https://github.com/huggingface/peft
[19]: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
[20]: https://scholar.google.com/scholar?oi=bibs&hl=en&authuser=2&cites=15052937328817631261,4097184
744846514103,1520777361382155671,17476825572045927382,18443729326628441434,14801318227356878622,7890
865700763267262,12854182577605049984,15641002901115500560,5104500764547628290
[21]: /EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
[22]: /EleutherAI/lm-evaluation-harness/blob/main/docs/config_files.md
[23]: /EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md
[24]: /EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md
[25]: https://huggingface.co/models
[26]: https://github.com/huggingface/accelerate
[27]: https://huggingface.co/docs/transformers/v4.15.0/en/parallelism
[28]: https://github.com/EleutherAI/gpt-neox/blob/main/eval.py
[29]: https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py
[30]: https://github.com/NVIDIA/NeMo
[31]: https://github.com/NVIDIA/NeMo?tab=readme-ov-file#installation
[32]: https://github.com/NVIDIA/NeMo/releases
[33]: https://github.com/EleutherAI/lm-evaluation-harness/tree/main?tab=readme-ov-file#install
[34]: https://catalog.ngc.nvidia.com/models
[35]: https://huggingface.co/nvidia
[36]: https://github.com/NVIDIA/NeMo/tree/main/scripts/nlp_language_modeling
[37]: https://docs.vllm.ai/en/latest/models/supported_models.html
[38]: https://github.com/EleutherAI/lm-evaluation-harness/blob/e74ec966556253fbe3d8ecba9de675c77c075
bce/lm_eval/models/vllm_causallms.py
[39]: /EleutherAI/lm-evaluation-harness/blob/main/scripts/model_comparator.py
[40]: https://docs.sglang.ai/index.html
[41]: https://docs.sglang.io/get_started/install.html#install-sglang
[42]: https://docs.flashinfer.ai/
[43]: /EleutherAI/lm-evaluation-harness/blob/main/pyproject.toml
[44]: https://docs.sglang.io/advanced_features/server_arguments.html
[45]: https://platform.openai.com/docs/guides/gpt
[46]: https://docs.anthropic.com/claude/reference/selecting-a-model
[47]: https://docs.anthropic.com/claude/docs/models-overview
[48]: https://textsynth.com/documentation.html#engines
[49]: https://github.com/EleutherAI/lm-evaluation-harness/pull/395
[50]: https://docs.cohere.com/docs/models
[51]: https://github.com/ggerganov/llama.cpp
[52]: https://github.com/abetlen/llama-cpp-python
[53]: https://github.com/ggerganov/llama.cpp
[54]: https://docs.vllm.ai/en/latest/models/supported_models.html
[55]: https://huggingface.co/state-spaces
[56]: https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2
[57]: https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/core/core.html#nemo-models
[58]: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx
[59]: /EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md
[60]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md#interface
[61]: https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py
[62]: https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md
[63]: https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py
[64]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-libra
ry-usage
[65]: https://github.com/huggingface/peft
[66]: https://github.com/ModelCloud/GPTQModel
[67]: https://github.com/PanQiWei/AutoGPTQ
[68]: https://huggingface.co/datasets/KonradSzafer/lm-eval-results-demo
[69]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
[70]: https://zenoml.com
[71]: https://hub.zenoml.com
[72]: https://hub.zenoml.com/account
[73]: /EleutherAI/lm-evaluation-harness/blob/main/examples/visualize-zeno.ipynb
[74]: https://wandb.ai/site
[75]: https://wandb.ai/authorize
[76]: https://docs.wandb.ai/ref/python/init
[77]: /EleutherAI/lm-evaluation-harness/blob/main/examples/visualize-wandb.ipynb
[78]: https://github.com/EleutherAI/lm-evaluation-harness/issues
[79]: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs
[80]: /EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md
[81]: https://arxiv.org/abs/2005.14165
[82]: https://discord.gg/eleutherai
