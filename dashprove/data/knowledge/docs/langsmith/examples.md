# LangSmith Cookbook

[[Release Notes]][1] [[Python Downloads]][2] [[NPM Version]][3] [[JS Downloads]][4]

Welcome to the LangSmith Cookbook — your practical guide to mastering [LangSmith][5]. While our
[standard documentation][6] covers the basics, this repository delves into common patterns and some
real-world use-cases, empowering you to optimize your LLM applications further.

This repository is your practical guide to maximizing [LangSmith][7]. As a tool, LangSmith empowers
you to debug, evaluate, test, and improve your LLM applications continuously. These recipes present
real-world scenarios for you to adapt and implement.

**Your Input Matters**

Help us make the cookbook better! If there's a use-case we missed, or if you have insights to share,
please raise a GitHub issue (feel free to tag [Will][8]) or contact the LangChain development team.
Your expertise shapes this community.

## Tracing your code

Tracing allows for seamless debugging and improvement of your LLM applications. Here's how:

* [Tracing without LangChain][9]: learn to trace applications independent of LangChain using the
  Python SDK's @traceable decorator.
* [REST API][10]: get acquainted with the REST API's features for logging LLM and chat model runs,
  and understand nested runs. The run logging spec can be found in the [LangSmith SDK
  repository][11].
* [Customizing Run Names][12]: improve UI clarity by assigning bespoke names to LangSmith chain
  runs—includes examples for chains, lambda functions, and agents.
* [Tracing Nested Calls within Tools][13]: include all nested tool subcalls in a single trace by
  using `run_manager.get_child()` and passing to the child `callbacks`
* [Display Trace Links][14]: add trace links to your app to speed up development. This is useful
  when prototyping your application in its unique UI, since it lets you quickly see its execution
  flow, add feedback to a run, or add the run to a dataset.

## LangChain Hub

Efficiently manage your LLM components with the [LangChain Hub][15]. For dedicated documentation,
please see the [hub docs][16].

* [RetrievalQA Chain][17]: use prompts from the hub in an example RAG pipeline.
* [Prompt Versioning][18]: ensure deployment stability by selecting specific prompt versions over
  the 'latest'.
* [Runnable PromptTemplate][19]: streamline the process of saving prompts to the hub from the
  playground and integrating them into runnable chains.

## Testing & Evaluation

Test and benchmark your LLM systems using methods in these evaluation recipes:

### Python Examples

* [Prompt Iteration Walkthrough][20]: run regression tests to compare multiple prompts on 3 datasets

**Retrieval Augmented Generation (RAG)**

* [Q&A System Correctness][21]: evaluate your retrieval-augmented Q&A pipeline end-to-end on a
  dataset. Iterate, improve, and keep testing.
* [Evaluating Q&A Systems with Dynamic Data][22]: use evaluators that dereference a labels to handle
  data that changes over time.
* [RAG Evaluation using Fixed Sources][23]: evaluate the response component of a RAG
  (retrieval-augmented generation) pipeline by providing retrieved documents in the dataset
* [RAG evaluation with RAGAS][24]: evaluate RAG pipelines using the [RAGAS][25] framework. Covers
  metrics for both the generator AND retriever in both labeled and reference-free contexts (answer
  correctness, faithfulness, context relevancy, recall and precision).

**Chat Bots**

* [Chat Bot Evals using Simulated Users][26]: evaluate your chat bot using a simulated user. The
  user is given a task, and you score your assistant based on how well it helps without being
  breaking its instructions.
* [Single-turn evals][27]: Evaluate chatbots within multi-turn conversations by treating each data
  point as an individual dialogue turn. This guide shows how to set up a multi-turn conversation
  dataset and evaluate a simple chat bot on it.

**Extraction**

* [Evaluating an Extraction Chain][28]: measure the similarity between the extracted structured
  content and structured labels using LangChain's json evaluators.
* [Exact Match][29]: deterministic comparison of your system output against a reference label.

**Agents**

* [Evaluating an Agent's intermediate steps][30]: compare the sequence of actions taken by an agent
  to an expected trajectory to grade effective tool use.
* [Tool Selection][31]: Evaluate the precision of selected tools. Include an automated prompt writer
  to improve the tool descriptions based on failure cases.

**Multimodel**

* [Evaluating Multimodal Models][32]: benchmark a multimodal image classification chain

**Fundamentals**

* [Backtesting][33]: benchmark new versions of your production app using real inputs. Convert
  production runs to a test dataset, then compare your new system's performance against the
  baseline.
* [Adding Metrics to Existing Tests][34]: Apply new evaluators to existing test results without
  re-running your model, using the `compute_test_metrics` utility function. This lets you evaluate
  "post-hoc" and backfill metrics as you define new evaluators.
* [Naming Test Projects][35]: manually name your tests with `run_on_dataset(...,
  project_name='my-project-name')`
* [Exporting Tests to CSV][36]: Use the `get_test_results` beta utility to easily export your test
  results to a CSV file. This allows you to analyze and report on the performance metrics, errors,
  runtime, inputs, outputs, and other details of your tests outside of the Langsmith platform.
* [How to download feedback and examples from a test project][37]: goes beyond the utility described
  above to query and export the predictions, evaluation results, and other information to
  programmatically add to your reports.

### TypeScript / JavaScript Testing Examples

Incorporate LangSmith into your TS/JS testing and evaluation workflow:

* [Vision-based Evals in JavaScript][38]: evaluate AI-generated UIs using GPT-4V

We are working to add more JS examples soon. In the meantime, check out the JS eval quickstart the
following guides:

* [JS LangSmith walkthrough][39]
* [Evaluation quickstart][40]

## Using Feedback

Harness user [feedback][41], "ai-assisted" feedback, and other signals to improve, monitor, and
personalize your applications. Feedback can be user-generated or "automated" using functions or even
calls to an LLM:

* [Streamlit Chat App][42]: a minimal chat app that captures user feedback and shares traces of the
  chat application.
  
  * The [vanilla_chain.py][43] contains an LLMChain that powers the chat application.
  * The [expression_chain.py][44] contains an equivalent chat chain defined exclusively with
    [LangChain expressions][45].
* [Next.js Chat App][46]: explore a simple TypeScript chat app demonstrating tracing and feedback
  capture.
  
  * You can [check out a deployed demo version here][47].
* [Building an Algorithmic Feedback Pipeline][48]: automate feedback metrics for advanced monitoring
  and performance tuning. This lets you evaluate production runs as a batch job.
* [Real-time Automated Feedback][49]: automatically generate feedback metrics for every run using an
  async callback. This lets you evaluate production runs in real-time.
* [Real-time RAG Chat Bot Evaluation][50]: This Streamlit walkthrough showcases an advanced
  application of the concepts from the [Real-time Automated Feedback][51] tutorial. It demonstrates
  how to automatically check for hallucinations in your RAG chat bot responses against the retrieved
  documents. For more information on RAG, [check out the LangChain docs][52].
* [LangChain Agents with LangSmith][53] instrument a LangChain web-search agent with tracing and
  human feedback.

## Optimization

Use LangSmith to help optimize your LLM systems, so they can continuously learn and improve.

* [Prompt Bootstrapping][54]: Optimize your prompt over a set of examples by incorporating human
  feedback and an LLM prompt optimizer. Works by rewriting an optimized system prompt based on
  feedback.
  
  * [Prompt Bootstrapping for style transfer: Elvis-Bot][55]: Extend prompt bootstrapping to
    generate outputs in the style of a specific persona. This notebook demonstrates how to create an
    "Elvis-bot" that mimics the tweet style of @omarsar0 by iteratively refining a prompt using
    Claude's exceptional prompt engineering capabilities and feedback collected through LangSmith's
    annotation queue.
* [Automated Few-shot Prompt Bootstrapping][56]: Automatically curate the most informative few-shot
  examples based on performance metrics, removing the need for manual example engineering. Applied
  to an entailment task on the SCONE dataset.
* [Iterative Prompt Optimization][57]: Streamlit app demonstrating real-time prompt optimization
  based on user feedback and dialog, leveraging few-shot learning and a separate "optimizer" model
  to dynamically improve a tweet-generating system.
* [Online Few-shot Examples][58] Configure online evaluators to add good examples to a dataset.
  Review, then use them as few-shot examples to boost performance.

## Exporting data for fine-tuning

Fine-tune an LLM on collected run data using these recipes:

* [OpenAI Fine-Tuning][59]: list LLM runs and convert them to OpenAI's fine-tuning format
  efficiently.
* [Lilac Dataset Curation][60]: further curate your LangSmith datasets using Lilac to detect
  near-duplicates, check for PII, and more.

## Exploratory Data Analysis

Turn your trace data into actionable insights:

* [Exporting LLM Runs and Feedback][61]: extract and interpret LangSmith LLM run data, making them
  ready for various analytical platforms.
* [Lilac][62]: enrich datasets using the open-source analytics tool, [Lilac][63], to better label
  and organize your data.

[1]: https://github.com/langchain-ai/langsmith-sdk/releases
[2]: https://pepy.tech/project/langsmith
[3]: https://camo.githubusercontent.com/710facc59f3810ad4abdb266464a80c9d5fbfe2987034283570164891dd3
5ac5/68747470733a2f2f696d672e736869656c64732e696f2f6e706d2f762f6c616e67736d6974683f6c6f676f3d6e706d
[4]: https://www.npmjs.com/package/langsmith
[5]: https://smith.langchain.com/
[6]: https://docs.smith.langchain.com/
[7]: https://smith.langchain.com/
[8]: https://github.com/hinthornw
[9]: /langchain-ai/langsmith-cookbook/blob/main/tracing-examples/traceable/tracing_without_langchain
.ipynb
[10]: /langchain-ai/langsmith-cookbook/blob/main/tracing-examples/rest/rest.ipynb
[11]: https://github.com/langchain-ai/langsmith-sdk/blob/main/openapi/openapi.yaml
[12]: /langchain-ai/langsmith-cookbook/blob/main/tracing-examples/runnable-naming/run-naming.ipynb
[13]: /langchain-ai/langsmith-cookbook/blob/main/tracing-examples/nesting-tools/nest_runs_within_too
ls.ipynb
[14]: /langchain-ai/langsmith-cookbook/blob/main/tracing-examples/show-trace-url-streamlit/README.md
[15]: https://smith.langchain.com/hub
[16]: https://docs.smith.langchain.com/hub/quickstart
[17]: /langchain-ai/langsmith-cookbook/blob/main/hub-examples/retrieval-qa-chain/retrieval-qa.ipynb
[18]: /langchain-ai/langsmith-cookbook/blob/main/hub-examples/retrieval-qa-chain-versioned/prompt-ve
rsioning.ipynb
[19]: /langchain-ai/langsmith-cookbook/blob/main/hub-examples/runnable-prompt/edit-in-playground.ipy
nb
[20]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/movie-demo/prompt_iteration.ipynb
[21]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/qa-correctness/qa-correctness.ipyn
b
[22]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/dynamic-data/testing_dynamic_data.
ipynb
[23]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/using-fixed-sources/using_fixed_so
urces.ipynb
[24]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/ragas/ragas.ipynb
[25]: https://docs.ragas.io/en/stable/
[26]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/chatbot-simulation/chatbot-simulat
ion.ipynb
[27]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/chat-single-turn/chat_evaluation_s
ingle_turn.ipynb
[28]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/data-extraction/contract-extractio
n.ipynb
[29]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/exact-match/exact_match.ipynb
[30]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/agent_steps/evaluating_agents.ipyn
b
[31]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/tool-selection/tool-selection.ipyn
b
[32]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/multimodal/multimodal.ipynb
[33]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/backtesting/backtesting.ipynb
[34]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/evaluate-existing-test-project/eva
luate_runs.ipynb
[35]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/naming-test-projects/naming-test-p
rojects.md
[36]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/export-test-to-csv/export-test-to-
csv.ipynb
[37]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/download-feedback-and-examples/dow
nload_example.ipynb
[38]: /langchain-ai/langsmith-cookbook/blob/main/typescript-testing-examples/vision-evals/vision-eva
ls.ipynb
[39]: https://js.langchain.com/docs/guides/langsmith_evaluation
[40]: https://docs.smith.langchain.com/evaluation/quickstart
[41]: https://docs.smith.langchain.com/tracing/faq/logging_feedback
[42]: /langchain-ai/langsmith-cookbook/blob/main/feedback-examples/streamlit/README.md
[43]: /langchain-ai/langsmith-cookbook/blob/main/feedback-examples/streamlit/vanilla_chain.py
[44]: /langchain-ai/langsmith-cookbook/blob/main/feedback-examples/streamlit/expression_chain.py
[45]: https://python.langchain.com/docs/expression_language/
[46]: /langchain-ai/langsmith-cookbook/blob/main/feedback-examples/nextjs/README.md
[47]: https://langsmith-cookbook.vercel.app/
[48]: /langchain-ai/langsmith-cookbook/blob/main/feedback-examples/algorithmic-feedback/algorithmic_
feedback.ipynb
[49]: /langchain-ai/langsmith-cookbook/blob/main/feedback-examples/realtime-algorithmic-feedback/rea
ltime_feedback.ipynb
[50]: /langchain-ai/langsmith-cookbook/blob/main/feedback-examples/streamlit-realtime-feedback/READM
E.md
[51]: /langchain-ai/langsmith-cookbook/blob/main/feedback-examples/realtime-algorithmic-feedback/rea
ltime_feedback.ipynb
[52]: https://python.langchain.com/docs/use_cases/question_answering/
[53]: /langchain-ai/langsmith-cookbook/blob/main/feedback-examples/streamlit-agent/README.md
[54]: /langchain-ai/langsmith-cookbook/blob/main/optimization/assisted-prompt-bootstrapping/assisted
-prompt-engineering.ipynb
[55]: /langchain-ai/langsmith-cookbook/blob/main/optimization/assisted-prompt-bootstrapping/elvis-bo
t.ipynb
[56]: /langchain-ai/langsmith-cookbook/blob/main/optimization/bootstrap-fewshot/bootstrap-few-shot.i
pynb
[57]: https://github.com/langchain-ai/tweet-critic
[58]: /langchain-ai/langsmith-cookbook/blob/main/testing-examples/movie-demo/optimization.ipynb
[59]: /langchain-ai/langsmith-cookbook/blob/main/fine-tuning-examples/export-to-openai/fine-tuning-o
n-chat-runs.ipynb
[60]: /langchain-ai/langsmith-cookbook/blob/main/fine-tuning-examples/lilac/lilac.ipynb
[61]: /langchain-ai/langsmith-cookbook/blob/main/exploratory-data-analysis/exporting-llm-runs-and-fe
edback/llm_run_etl.ipynb
[62]: /langchain-ai/langsmith-cookbook/blob/main/exploratory-data-analysis/lilac/lilac.ipynb
[63]: https://github.com/lilacai/lilac
