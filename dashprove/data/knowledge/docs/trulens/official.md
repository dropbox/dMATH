[TruLens is now OpenTelemetry Compatible 🎉 Learn how to leverage it for agents! ][1]

TruLens: Evals and Tracing for Agents
[ ][2]

* [
  
  Blog
  
  ][3]
* [
  
  GitHub
  
  ][4]
* [
  
  Community
  
  ][5]
* [
  Documentation
  ][6]

# Evaluate and Trace AI Agents

Evaluate, iterate faster, and select your best AI agent with TruLens.

[
GIVE US A STAR ⭐️
][7]
[ [PyPI - Version] ][8] [ [Azure DevOps builds (job)] ][9] [ [GitHub] ][10] [ [PyPI - Downloads]
][11] [ [Discourse Community] ][12] [ [Docs] ][13] [ [Open In Colab] ][14]

## TruLens: Move from Vibes to Metrics

Ship agentic workflows to production, faster. TruLens helps you objectively measure the quality and
effectiveness of your agent using feedback functions. Feedback functions help to programmatically
evaluate the critical components of your app's execution flow, such as retrieved context, tool calls
and plans, so that you can expedite and scale up experiment evaluation. Use it for a wide variety of
use cases including agents,retrieval-augmented generation, summarization and more.

## Trusted by

[ [Equinix] ][15] [ [tribble.ai] ][16] [ [KBC Group] ][17] [ [Snowflake] ][18] [ [CubeServ] ][19] [
[Datec] ][20]

### Evaluate

Evaluate how your choices are performing across multiple feedback functions, such as:

* Groundedness
* Context Relevance
* Coherence

### Iterate

Leverage and add to an extensible library of built-in feedback functions. Observe where apps have
weaknesses to inform iteration on prompts, hyperparameters, and more.

### Test

Compare different LLM apps on a metrics leaderboard to pick the best performing one.

## How it works

[TruLens diagram] [TruLens diagram vertical]

## Why Use TruLens to Validate Your AI Agent?

The fastest, easiest way to validate your AI Agent.

### Interoperable tracing.

TruLens emits and evaluates OpenTelemetry traces, making it easy to integrate with your existing
observability stack.

### Scalable, trusted evals.

TruLens provides trusted, benchmarked evals to evaluate your agent's performance. Read more about
our [benchmarks][21] and [optimization process][22].

### Get the breadth of feedback you need to evaluate app performance.

TruLens evaluates AI agents with feedback functions to measure their performance and minimize risk:

* Context Relevance
* Groundedness
* Answer Relevance
* Comprehensiveness
* Harmful or toxic language
* User sentiment
* Language mismatch
* Fairness and bias
* Or other custom feedback functions you provide

## TruLens can work with any AI Agent

Use TruLens for any AI Agent via the Python SDK or by ingesting OpenTelemetry traces.

* ### TruLens is loved by thousands of users for applications such as:
  
  Agents
  
  Retrieval Augmented Generation (RAG)
  
  Summarization
  
  Co-pilots
* ### Use TruLens to identify the best performing version of your agent:
  
  Quickly compare metrics across versions and identify trace-level regressions.
  
  Make informed trade-offs between accuracy, reliability, cost, and latency.
  
  See how the execution flow of your agent changes across versions.

## Get started using TruLens today

You are critical to the ongoing success of TruLens. We encourage you to get started and provide
ample feedback, so that TruLens improves over time.

[ ][23]

Download

Get started with [pip install trulens][24].
[ ][25]

Documentation

Read about the library [here][26].
[ ][27]

Community

Come join the TruLens community on the [AI R&D Discourse Forum][28].

### What's a Feedback Function?

A feedback function scores the output of an LLM application by analyzing generated text from an LLM
(or a downstream model or application built on it) and metadata.

This is similar to labeling functions. A human-in-the-loop can be used to discover a relationship
between the feedback and input text. By modeling this relationship, we can then programmatically
apply it to scale up model evaluation. You can read more in this blog: ["What's Missing to Evaluate
Foundation Models at Scale"][29]

## TruLens is shepherded by Snowflake

Originally created by TruEra, TruLens is a community-driven open source project used by thousands of
developers to make credible LLM apps faster. Since TruEra's acquisition by Snowflake, Snowflake now
actively oversees and supports the development of TruLens in open source. [Read more about
Snowflake's commitment to growing TruLens in open source.][30]

Why a colossal squid?

The colossal squid's eyeball is about the size of a soccer ball, making it the largest eyeball of
any living creature. In addition, did you know that its eyeball contains light organs? That means
that colossal squids have automatic headlights when looking around. We're hoping to bring similar
guidance to model developers when creating, introspecting, and debugging neural networks. Read more
about the [amazing eyes of the colossal squid][31].

TruLens 2025

[1]: ../blog/otel_for_the_agentic_world
[2]: https://trulens.org
[3]: /blog/
[4]: https://github.com/truera/trulens/
[5]: https://snowflake.discourse.group/c/ai-research-and-development-community/trulens/97
[6]: getting_started
[7]: https://github.com/truera/trulens/
[8]: https://pypi.org/project/trulens/
[9]: https://dev.azure.com/truera/TruLens/_build/latest?definitionId=9&branchName=master
[10]: https://github.com/truera/trulens
[11]: https://pypi.org/project/trulens/
[12]: https://snowflake.discourse.group/c/ai-research-and-development-community/trulens/97
[13]: getting_started
[14]: https://colab.research.google.com/github/truera/trulens/blob/main/examples/quickstart/quicksta
rt.ipynb
[15]: https://equinix.com
[16]: https://tribble.ai/
[17]: https://www.kbc.com/
[18]: https://snowflake.com
[19]: https://www.cubeserv.com/
[20]: https://www.datec.com.bo/
[21]: https://www.snowflake.com/en/engineering-blog/benchmarking-LLM-as-a-judge-RAG-triad-metrics/
[22]: https://www.snowflake.com/en/engineering-blog/eval-guided-optimization-llm-judges-rag-triad/
[23]: https://pypi.org/project/trulens/
[24]: https://pypi.org/project/trulens/
[25]: https://www.trulens.org/getting_started/
[26]: trulens/install/
[27]: https://snowflake.discourse.group/c/ai-research-and-development-community/trulens/97
[28]: https://snowflake.discourse.group/c/ai-research-and-development-community/trulens/97
[29]: https://truera.com/whats-missing-to-evaluate-foundation-models-at-scale/
[30]: https://www.snowflake.com/en/blog/trulens-open-source-ai/
[31]: https://www.tepapa.govt.nz/discover-collections/read-watch-play/science/anatomy-colossal-squid
/eyes-colossal-squid
