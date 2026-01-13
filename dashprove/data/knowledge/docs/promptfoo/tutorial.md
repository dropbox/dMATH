* Getting Started
On this page

# Getting started

After [installing][1] promptfoo, you can set up your first config file in two ways:

## Running the example[​][2]

Set up your first config file with a pre-built example by running this command with [npx][3],
[npm][4], or [brew][5]:

* npx
* npm
* brew
`npx promptfoo@latest init --example getting-started
`
`npm install -g promptfoo 
promptfoo init --example getting-started
`
`brew install promptfoo
promptfoo init --example getting-started
`

This will create a new directory with a basic example that tests translation prompts across
different models. The example includes:

* A configuration file `promptfooconfig.yaml` with sample prompts, providers, and test cases.
* A `README.md` file explaining how the example works.

## Starting from scratch[​][6]

If you prefer to start from scratch instead of using the example, simply run `promptfoo init`
without the `--example` flag:

* npx
* npm
* brew
`npx promptfoo@latest init
`
`promptfoo init
`
`promptfoo init
`

The command will guide you through an interactive setup process to create your custom configuration.

## Configuration[​][7]

To configure your evaluation:

1. **Set up your prompts**: Open `promptfooconfig.yaml` and add prompts that you want to test. Use
   double curly braces for variable placeholders: `{{variable_name}}`. For example:
   
   `prompts:
     - 'Convert this English to {{language}}: {{input}}'
     - 'Translate to {{language}}: {{input}}'
   `
   
   [» More information on setting up prompts][8]
2. Add `providers` to specify AI models you want to test. Promptfoo supports 50+ providers including
   OpenAI, Anthropic, Google, and many others:
   
   `providers:
     - openai:gpt-5
     - openai:gpt-5-mini
     - anthropic:messages:claude-sonnet-4-5-20250929
     - vertex:gemini-2.5-pro
     # Or use your own custom provider
     - file://path/to/custom/provider.py
   `
   
   Each provider is specified using a simple format: `provider_name:model_name`. For example:
   
   * `openai:gpt-5` for GPT-5
   * `openai:gpt-5-mini` for OpenAI's GPT-5 Mini
   * `anthropic:messages:claude-sonnet-4-5-20250929` for Anthropic's Claude
   * `bedrock:us.meta.llama3-3-70b-instruct-v1:0` for Meta's Llama 3.3 70B via AWS Bedrock
   
   Most providers need authentication. For OpenAI:
   
   `export OPENAI_API_KEY=sk-abc123
   `
   
   You can use:
   
   * **Cloud APIs**: [OpenAI][9], [Anthropic][10], [Google][11], [Mistral][12], and [many more][13]
   * **Local Models**: [Ollama][14], [llama.cpp][15], [LocalAI][16]
   * **Custom Code**: [Python][17], [JavaScript][18], or any [executable][19]
   
   [» See our full providers documentation][20] for detailed setup instructions for each provider.
3. **Add test inputs**: Add some example inputs for your prompts. Optionally, add [assertions][21]
   to set output requirements that are checked automatically.
   
   For example:
   
   `tests:
     - vars:
         language: French
         input: Hello world
     - vars:
         language: Spanish
         input: Where is the library?
   `
   
   When writing test cases, think of core use cases and potential failures that you want to make
   sure your prompts handle correctly.
   
   [» More information on setting up tests][22]
4. **Run the evaluation**: Make sure you're in the directory containing `promptfooconfig.yaml`, then
   run:
   
   * npx
   * npm
   * brew
   `npx promptfoo@latest eval
   `
   `promptfoo eval
   `
   `promptfoo eval
   `
   
   This tests every prompt, model, and test case.
5. After the evaluation is complete, open the web viewer to review the outputs:
   
   * npx
   * npm
   * brew
   `npx promptfoo@latest view
   `
   `promptfoo view
   `
   `promptfoo view
   `

[Promptfoo Web UI showing evaluation results]

### Configuration[​][23]

The YAML configuration format runs each prompt through a series of example inputs (aka "test case")
and checks if they meet requirements (aka "assert").

Asserts are *optional*. Many people get value out of reviewing outputs manually, and the web UI
helps facilitate this.

tip

See the [Configuration docs][24] for a detailed guide.

Show example YAML
`# yaml-language-server: $schema=https://promptfoo.dev/config-schema.json
description: Automatic response evaluation using LLM rubric scoring

# Load prompts
prompts:
  - file://prompts.txt
providers:
  - openai:gpt-5
defaultTest:
  assert:
    - type: llm-rubric
      value: Do not mention that you are an AI or chat assistant
    - type: javascript
      # Shorter is better
      value: Math.max(0, Math.min(1, 1 - (output.length - 100) / 900));
tests:
  - vars:
      name: Bob
      question: Can you help me find a specific product on your website?
  - vars:
      name: Jane
      question: Do you have any promotions or discounts currently available?
  - vars:
      name: Ben
      question: Can you check the availability of a product at a specific store location?
  - vars:
      name: Dave
      question: What are your shipping and return policies?
  - vars:
      name: Jim
      question: Can you provide more information about the product specifications or features?
  - vars:
      name: Alice
      question: Can you recommend products that are similar to what I've been looking at?
  - vars:
      name: Sophie
      question: Do you have any recommendations for products that are currently popular or trending?
  - vars:
      name: Jessie
      question: How can I track my order after it has been shipped?
  - vars:
      name: Kim
      question: What payment methods do you accept?
  - vars:
      name: Emily
      question: Can you help me with a problem I'm having with my account or order?
`

## Examples[​][25]

### Prompt quality[​][26]

In [this example][27], we evaluate whether adding adjectives to the personality of an assistant bot
affects the responses.

You can quickly set up this example by running:

* npx
* npm
* brew
`npx promptfoo@latest init --example self-grading
`
`promptfoo init --example self-grading
`
`promptfoo init --example self-grading
`

Here is the configuration:

promptfooconfig.yaml
`# yaml-language-server: $schema=https://promptfoo.dev/config-schema.json
# Load prompts
prompts:
  - file://prompts.txt

# Set an LLM
providers:
  - openai:gpt-5

# These test properties are applied to every test
defaultTest:
  assert:
    # Ensure the assistant doesn't mention being an AI
    - type: llm-rubric
      value: Do not mention that you are an AI or chat assistant

    # Prefer shorter outputs using a scoring function
    - type: javascript
      value: Math.max(0, Math.min(1, 1 - (output.length - 100) / 900));

# Set up individual test cases
tests:
  - vars:
      name: Bob
      question: Can you help me find a specific product on your website?
  - vars:
      name: Jane
      question: Do you have any promotions or discounts currently available?
  - vars:
      name: Ben
      question: Can you check the availability of a product at a specific store location?
  - vars:
      name: Dave
      question: What are your shipping and return policies?
  - vars:
      name: Jim
      question: Can you provide more information about the product specifications or features?
  - vars:
      name: Alice
      question: Can you recommend products that are similar to what I've been looking at?
  - vars:
      name: Sophie
      question: Do you have any recommendations for products that are currently popular or trending?
  - vars:
      name: Jessie
      question: How can I track my order after it has been shipped?
  - vars:
      name: Kim
      question: What payment methods do you accept?
  - vars:
      name: Emily
      question: Can you help me with a problem I'm having with my account or order?
`

A simple `npx promptfoo@latest eval` will run this example from the command line:

[promptfoo command line]

This command will evaluate the prompts, substituting variable values, and output the results in your
terminal.

Have a look at the setup and full output [here][28].

You can also output a nice [spreadsheet][29], [JSON][30], YAML, or an HTML file:

* npx
* npm
* brew
`npx promptfoo@latest eval -o output.html
`
`promptfoo eval -o output.html
`
`promptfoo eval -o output.html
`

[Table output]

### Model quality[​][31]

In [this next example][32], we evaluate the difference between GPT-5 and GPT-5 Mini outputs for a
given prompt:

You can quickly set up this example by running:

* npx
* npm
* brew
`npx promptfoo@latest init --example openai-model-comparison
`
`promptfoo init --example openai-model-comparison
`
`promptfoo init --example openai-model-comparison
`
promptfooconfig.yaml
`prompts:
  - file://prompt1.txt
  - file://prompt2.txt

# Set the LLMs we want to test
providers:
  - openai:gpt-5-mini
  - openai:gpt-5
`

A simple `npx promptfoo@latest eval` will run the example. Also note that you can override
parameters directly from the command line. For example, this command:

* npx
* npm
* brew
`npx promptfoo@latest eval -p prompts.txt -r openai:gpt-5-mini openai:gpt-5 -o output.html
`
`promptfoo eval -p prompts.txt -r openai:gpt-5-mini openai:gpt-5 -o output.html
`
`promptfoo eval -p prompts.txt -r openai:gpt-5-mini openai:gpt-5 -o output.html
`

Produces this HTML table:

[Side-by-side eval of LLM model quality, gpt-5 vs gpt-5-mini, html output]

Full setup and output [here][33].

A similar approach can be used to run other model comparisons. For example, you can:

* Compare same models with different temperatures (see [GPT temperature comparison][34])
* Compare Llama vs. GPT (see [Llama vs GPT benchmark][35])
* Compare Retrieval-Augmented Generation (RAG) with LangChain vs. regular GPT-4 (see [LangChain
  example][36])

## Additional Resources[​][37]

* [» Configuration guide][38] for detailed setup instructions
* [» Providers documentation][39] for all supported AI models and services
* [» Assertions & Metrics][40] for automatically assessing outputs

## More Examples[​][41]

* There are many examples available in the [`examples/`][42] directory of our Github repository.

## Automatically assess outputs[​][43]

The above [examples][44] create a table of outputs that can be manually reviewed. By setting up
assertions, you can automatically grade outputs on a pass/fail basis.

For more information on automatically assessing outputs, see [Assertions & Metrics][45].

[Edit this page][46]
Last updated on Dec 22, 2025 by dependabot[bot]

[1]: /docs/installation/
[2]: #running-the-example
[3]: https://nodejs.org/en/download
[4]: https://nodejs.org/en/download
[5]: https://brew.sh/
[6]: #starting-from-scratch
[7]: #configuration
[8]: /docs/configuration/prompts/
[9]: /docs/providers/openai/
[10]: /docs/providers/anthropic/
[11]: /docs/providers/google/
[12]: /docs/providers/mistral/
[13]: /docs/providers/
[14]: /docs/providers/ollama/
[15]: /docs/providers/llama.cpp/
[16]: /docs/providers/localai/
[17]: /docs/providers/python/
[18]: /docs/providers/custom-api/
[19]: /docs/providers/custom-script/
[20]: /docs/providers/
[21]: /docs/configuration/expected-outputs/
[22]: /docs/configuration/guide/
[23]: #configuration-1
[24]: /docs/configuration/guide/
[25]: #examples
[26]: #prompt-quality
[27]: https://github.com/promptfoo/promptfoo/tree/main/examples/self-grading
[28]: https://github.com/promptfoo/promptfoo/tree/main/examples/self-grading
[29]: https://docs.google.com/spreadsheets/d/1nanoj3_TniWrDl1Sj-qYqIMD6jwm5FBy15xPFdUTsmI/edit?usp=s
haring
[30]: https://github.com/promptfoo/promptfoo/blob/main/examples/simple-cli/output.json
[31]: #model-quality
[32]: https://github.com/promptfoo/promptfoo/tree/main/examples/openai-model-comparison
[33]: https://github.com/promptfoo/promptfoo/tree/main/examples/openai-model-comparison
[34]: https://github.com/promptfoo/promptfoo/tree/main/examples/gpt-4o-temperature-comparison
[35]: /docs/guides/compare-llama2-vs-gpt/
[36]: /docs/configuration/testing-llm-chains/
[37]: #additional-resources
[38]: /docs/configuration/guide/
[39]: /docs/providers/
[40]: /docs/configuration/expected-outputs/
[41]: #more-examples
[42]: https://github.com/promptfoo/promptfoo/tree/main/examples
[43]: #automatically-assess-outputs
[44]: https://github.com/promptfoo/promptfoo/tree/main/examples
[45]: /docs/configuration/expected-outputs/
[46]: https://github.com/promptfoo/promptfoo/tree/main/site/docs/getting-started.md
