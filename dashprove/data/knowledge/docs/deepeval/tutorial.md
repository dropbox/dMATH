* Getting Started
* Quick Introduction
On this page

# Quick Introduction

**DeepEval** is an open-source evaluation framework for LLMs. DeepEval makes it extremely easy to
build and iterate on LLM (applications) and was built with the following principles in mind:

* Easily "unit test" LLM outputs in a similar way to Pytest.
* Plug-and-use 50+ LLM-evaluated metrics, most with research backing, and all are multi-modal.
* Evaluation for RAG, agents, chatbots, and virtually any use case.
* Supports both end-to-end and component level evaluation.
* Synthetic dataset generation with state-of-the-art evolution techniques.
* Metrics are simple to customize and covers all use cases.
* Red team, safety scan LLM applications for security vulnerabilities.

Additionally, DeepEval has a cloud platform [Confident AI][1], which allow teams to use DeepEval to
**evaluate, regression test, red team, and monitor** LLM applications on the cloud.

Delivered by
Confident AI

## Installation[​][2]

In a newly created virtual environment, run:

`pip install -U deepeval
`

`deepeval` runs evaluations locally on your environment. To keep your testing reports in a
centralized place on the cloud, use [Confident AI][3], the native evaluation platform for DeepEval:

`deepeval login
`
Configure Environment Variables

DeepEval autoloads environment files (at import time)

* **Precedence:** existing process env -> `.env.local` -> `.env`
* **Opt-out:** set `DEEPEVAL_DISABLE_DOTENV=1`

More information on `env` settings can be [found here.][4]

`# quickstart
cp .env.example .env.local
# then edit .env.local (ignored by git)
`
note

Confident AI is free and allows you to keep all evaluation results on the cloud. Sign up [here.][5]

## Create Your First Test Run[​][6]

Create a test file to run your first **end-to-end evaluation**.

* Single-Turn
* Multi-Turn

An [LLM test case][7] in `deepeval` represents a **single unit of LLM app interaction**, and
contains mandatory fields such as the `input` and `actual_output` (LLM generated output), and
optional ones like `expected_output`.

[LLM Test Case]

Run `touch test_example.py` in your terminal and paste in the following code:

test_example.py
`from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        # Replace this with the actual output from your LLM application
        actual_output="A persistent cough and fever could be a viral infection or something more ser
ious. See a doctor if symptoms worsen or don't improve in a few days.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mi
ld viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical at
tention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty 
breathing, chest pain, or other concerning signs."
    )
    assert_test(test_case, [correctness_metric])
`

Then, run `deepeval test run` from the root directory of your project to evaluate your LLM app
**end-to-end**:

`deepeval test run test_example.py
`

Congratulations! Your test case should have passed ✅ Let's breakdown what happened.

* The variable `input` mimics a user input, and `actual_output` is a placeholder for what your
  application's supposed to output based on this input.
* The variable `expected_output` represents the ideal answer for a given `input`, and [`GEval`][8]
  is a research-backed metric provided by `deepeval` for you to evaluate your LLM output's on any
  custom metric with human-like accuracy.
* In this example, the metric `criteria` is correctness of the `actual_output` based on the provided
  `expected_output`, but not all metrics require an `expected_output`.
* All metric scores range from 0 - 1, which the `threshold=0.5` threshold ultimately determines if
  your test have passed or not.

If you run more than one test run, you will be able to **catch regressions** by comparing test cases
side-by-side. This is also made easier if you're using `deepeval` alongside Confident AI ([see
below][9] for video demo).

A [conversational test case][10] in `deepeval` represents a **multi-turn interaction with your LLM
app**, and contains information such as the actual conversation that took place in the format of
`turn`s, and optionally the scenario of which a conversation happened.

[Conversational Test Case]

Run `touch test_example.py` in your terminal and paste in the following code:

test_example.py
`from deepeval import assert_test
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationalGEval

def test_professionalism():
    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant has acted professionally based on the content.",
        threshold=0.5
    )
    test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="What is DeepEval?"),
            Turn(role="assistant", content="DeepEval is an open-source LLM eval package.")
        ]
    )
    assert_test(test_case, [professionalism_metric])
`

Then, run `deepeval test run` from the root directory of your project to evaluate your LLM app
**end-to-end**:

`deepeval test run test_example.py
`

🎉 Congratulations! Your test case should have passed ✅ Let's breakdown what happened.

* The variable `role` distinguishes between the end user and your LLM application, and `content`
  contains either the user’s input or the LLM’s output.
* In this example, the `criteria` metric evaluates the professionalism of the sequence of `content`.
* All metric scores range from 0 - 1, which the `threshold=0.5` threshold ultimately determines if
  your test have passed or not.

If you run more than one test run, you will be able to **catch regressions** by comparing test cases
side-by-side. This is also made easier if you're using `deepeval` alongside Confident AI ([see
below][11] for video demo).

info

Since almost all `deepeval` metrics including `GEval` are LLM-as-a-Judge metrics, you'll need to set
your `OPENAI_API_KEY` as an env variable. You can also customize the model used for evals:

`correctness_metric = GEval(..., model="o1")
`

DeepEval also integrates with these model providers: [Ollama][12], [Azure OpenAI][13],
[Anthropic][14], [Gemini][15], etc. To use **ANY** custom LLM of your choice, [check out this part
of the docs][16].

Evaluations getting "stuck"?

Most likely your evaluation LLM is failing and this might be due to rate limits or insufficient
quotas. By default, `deepeval` retries **transient** LLM errors once (2 attempts total):

* **Retried:** network/timeout errors and **5xx** server errors.
* **Rate limits (429):** retried unless the provider marks them non-retryable (for OpenAI,
  `insufficient_quota` is treated as non-retryable).
* **Backoff:** exponential with jitter (initial **1s**, base **2**, jitter **2s**, cap **5s**).

You can tune these via environment flags (no code changes). See [environment variables][17] for
details.

### Save Results[​][18]

It is recommended that you manage your evaluation suite on Confident AI, the `deepeval` platform.

* Confident AI
* Locally in JSON

Confident AI is the `deepeval` cloud, and helps you build the best LLM evals pipeline. Run `deepeval
view` to view your newly ran test run on the platform:

`deepeval view
`

The `deepeval view` command requires that the test run that you ran above has been successfully
cached locally. If something errors, simply run a new test run after logging in with `deepeval
login`:

`deepeval login
`

After you've pasted in your API key, Confident AI will **generate testing reports and automate
regression testing** whenever you run a test run to evaluate your LLM application inside any
environment, at any scale, anywhere.

Watch Full Guide on Confident AI

**Once you've ran more than one test run**, you'll be able to use the [regression testing page][19]
shown near the end of the video. Green rows indicate that your LLM has shown improvement on specific
test cases, whereas red rows highlight areas of regression.

Simply set the `DEEPEVAL_RESULTS_FOLDER` environment variable to your relative path of choice.

`# linux
export DEEPEVAL_RESULTS_FOLDER="./data"

# or windows
set DEEPEVAL_RESULTS_FOLDER=.\data
`

## Test Runs With LLM Tracing[​][20]

While end-to-end evals treat your LLM app as a black-box, you also evaluate **individual
components** within your LLM app through **LLM tracing**. This is the recommended way to evaluate AI
agents.

[component level evals]

First paste in the following code:

main.py
`from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric

# 1. Decorate your app
@observe()
def llm_app(input: str):
  # 2. Decorate components with metrics you wish to evaluate or debug
  @observe(metrics=[AnswerRelevancyMetric()])
  def inner_component():
      # 3. Create test case at runtime
      update_current_span(test_case=LLMTestCase(input="Why is the blue sky?", actual_output="You mea
n why is the sky blue?"))

  return inner_component()

# 4. Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="Test input")])

# 5. Loop through dataset
for golden in dataset.evals_iterator():
  # 6. Call LLM app
  llm_app(golden.input)
`

Then run `python main.py` to run a **component-level** eval:

`python main.py
`

🎉 Congratulations! Your test case should have passed again ✅ Let's breakdown what happened.

* The `@observe` decorate tells `deepeval` where each component is and **creates an LLM trace** at
  execution time
* Any `metrics` supplied to `@observe` allows `deepeval` to evaluate that component based on the
  `LLMTestCase` you create
* In this example `AnswerRelevancyMetric()` was used to evaluate `inner_component()`
* The `dataset` specifies the **goldens** which will be used to invoke your `llm_app` during
  evaluation, which happens in a simple for loop

Once the for loop has ended, `deepeval` will aggregate all metrics, test cases in each component,
and run evals across them all, before generating the final testing report.

info

When you do LLM tracing using `deepeval`, you can automatically evals on **traces, spans, and
threads (conversations) in production**. Simply get an [API key from Confident AI][21] and set it in
the CLI:

`CONFIDENT_API_KEY="confident_us..."
`

`deepeval`'s LLM tracing implementation is **non-instrusive**, meaning it will not affect any part
of your code.

* Trace (end-to-end) Evals in Prod
* Span (component-level) Evals in Prod
* Thread (conversation) Evals in Prod

Evals on traces are [end-to-end evaluations][22], where a single LLM interaction is being evaluated.

Trace-Level Evals in Production

Spans make up a trace and evals on spans represents [component-level evaluations][23], where
individual components in your LLM app are being evaluated.

Span-Level Evals in Production

Threads are made up of **one or more traces**, and represents a multi-turn interaction to be
evaluated.

Thread (conversation) Evals in Production

## Continue With Your Use Case[​][24]

Tell us what you're building for more tailored onboarding:

[
AI Agents

* Setup LLM tracing
* Test end-to-end task completion
* Evaluate individual components
][25][
RAG

* Evaluate RAG end-to-end
* Test retriever and generator separately
* Multi-turn RAG evals
][26][
Chatbots

* Setup multi-turn test cases
* Evaluate turns in a conversation
* Simulate user interactions
][27]

**All quickstarts include a guide on how to bring evals to production near the end*

## Two Modes of LLM Evals[​][28]

`deepeval` offers two main modes of evaluation:

[
End-to-End LLM Evals

Best for: Raw LLM APIs, simple apps (no agents), chatbots, and occasionally RAG.

* Treats your LLM app as a black-box
* Minimal setup, unopinionated
* Can be included in CI/CD
* For single and multi-turn
][29][
Component-Level LLM Evals

Best for: AI agents, complex workflows, MCP evals, component-based RAG.

* Full visibility into your LLM app, white-box testing
* Setup non-instrusive LLM tracing
* Can be included in CI/CD
* Best for single-turn
][30]

## Essential Resources[​][31]

These are things you should definitely learn about:

[
Metrics

Learn about the 50+ metrics available, how to choose, and how to customize them.

][32][
Datasets

Learn how they are used within DeepEval, the concept of goldens, and how to use them for evals.

][33][
Tracing

Learn how to trace your LLM applications, evaluate on a component-level, and monitor in production.

][34]

## Other Products[​][35]

Learn more offerings available in `deepeval`'s ecosystem:

[
Confident AI

The cloud platform for DeepEval. Allow both technical and non-technical teams to collaborate on
testing AI, from evaluation in /dev to /prod.

][36][
DeepTeam

DeepTeam is DeepEval for AI safety and security testing. Expose 50+ vulnerabilities, with 20+ attack
methods such as tree jailbreaking all automated.

][37]

## Full Example[​][38]

You can find the full example [here on our Github][39].

[Edit this page][40]
Last updated on Dec 21, 2025 by Jeffrey Ip

[1]: https://app.confident-ai.com
[2]: #installation
[3]: https://www.confident-ai.com
[4]: /docs/evaluation-flags-and-configs#environment-flags
[5]: https://app.confident-ai.com
[6]: #create-your-first-test-run
[7]: /docs/evaluation-test-cases#llm-test-case
[8]: /docs/metrics-llm-evals
[9]: /docs/getting-started#save-results-on-cloud
[10]: /docs/evaluation-multiturn-test-cases#conversational-test-case
[11]: /docs/getting-started#save-results-on-cloud
[12]: https://deepeval.com/integrations/models/ollama
[13]: https://deepeval.com/integrations/models/azure-openai
[14]: https://deepeval.com/integrations/models/anthropic
[15]: https://deepeval.com/integrations/models/gemini
[16]: /guides/guides-using-custom-llms
[17]: /docs/environment-variables
[18]: #save-results
[19]: https://www.confident-ai.com/docs/llm-evaluation/dashboards/ab-regression-testing
[20]: #test-runs-with-llm-tracing
[21]: https://app.confident-ai.com
[22]: /docs/evaluation-end-to-end-llm-evals
[23]: /docs/evaluation-component-level-llm-evals
[24]: #continue-with-your-use-case
[25]: /docs/getting-started-agents
[26]: /docs/getting-started-rag
[27]: /docs/getting-started-chatbots
[28]: #two-modes-of-llm-evals
[29]: /docs/evaluation-end-to-end-llm-evals
[30]: /docs/evaluation-component-level-llm-evals
[31]: #essential-resources
[32]: /docs/metrics-introduction
[33]: /docs/evaluation-datasets
[34]: /docs/evaluation-llm-tracing
[35]: #other-products
[36]: https://www.confident-ai.com/docs
[37]: https://trydeepteam.com
[38]: #full-example
[39]: https://github.com/confident-ai/deepeval/blob/main/examples/getting_started/test_example.py
[40]: https://github.com/confident-ai/deepeval/edit/main/docs/docs/getting-started.mdx
