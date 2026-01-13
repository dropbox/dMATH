# Python API[#][1]

The primary way for using guardrails in your project is:

1. Create a [`RailsConfig`][2] object.
2. Create an [`LLMRails`][3] instance which provides an interface to the LLM that automatically
   applies the configured guardrails.
3. Generate LLM responses using the [`LLMRails.generate(...)`][4] or
   [`LLMRails.generate_async(...)`][5] methods.

## Basic usage[#][6]

from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("path/to/config")

app = LLMRails(config)
new_message = app.generate(messages=[{
    "role": "user",
    "content": "Hello! What can you do for me?"
}])

## RailsConfig[#][7]

The [`RailsConfig`][8] class contains the key bits of information for configuring guardrails:

* `models`: The list of models used by the rails configuration.
* `user_messages`: The list of user messages that should be used for the rails.
* `bot_messages`: The list of bot messages that should be used for the rails.
* `flows`: The list of flows that should be used for the rails.
* `instructions`: List of instructions in natural language (currently, only general instruction is
  supported).
* `docs`: List of documents included in the knowledge base.
* `sample_conversation`: The sample conversation to be used inside the prompts.
* `actions_server_url`: The actions server to be used. If specified, the actions will be executed
  through the actions server.

## Message Generation[#][9]

To use a guardrails configuration, you can call the [`LLMRails.generate`][10] or
[`LLMRails.generate_async`][11] methods.

The [`LLMRails.generate`][12] method takes as input either a `prompt` or a `messages` array. When a
prompt is provided, the guardrails apply as in a single-turn conversation. The structure of a
message is the following:

properties:
  role:
    type: "string"
    enum: ["user", "assistant", "context"]
  content:
    oneOf:
      - type: "string"
      - type: "object"

An example of conversation history is the following:

[
  {
    "role": "user",
    "content": "Hello!"
  },
  {
    "role": "assistant",
    "content": "Hello! How can I help you?"
  },
  {
    "role": "user",
    "content": "I want to know if my insurance covers certain expenses."
  }
]

An example which also sets the initial context is the following:

[
  {
    "role": "context",
    "content": {
      "user_name": "John",
      "access_level": "admin"
    }
  },
  {
    "role": "user",
    "content": "Hello!"
  },
  {
    "role": "assistant",
    "content": "Hello! How can I help you?"
  },
  {
    "role": "user",
    "content": "I want to know if my insurance covers certain expenses."
  }
]

## Actions[#][13]

Actions are a key component of the Guardrails toolkit. Actions enable the execution of python code
inside guardrails.

### Default Actions[#][14]

The following are the default actions included in the toolkit:

Core actions:

* `generate_user_intent`: Generate the canonical form for what the user said.
* `generate_next_step`: Generates the next step in the current conversation flow.
* `generate_bot_message`: Generate a bot message based on the desired bot intent.
* `retrieve_relevant_chunks`: Retrieves the relevant chunks from the knowledge base and adds them to
  the context.

Guardrail-specific actions:

* `self_check_facts`: Check the facts for the last bot response w.r.t. the extracted relevant chunks
  from the knowledge base.
* `self_check_input`: Check if the user input should be allowed.
* `self_check_output`: Check if the bot response should be allowed.
* `self_check_hallucination`: Check if the last bot response is a hallucination.

For convenience, this toolkit also includes a selection of LangChain tools, wrapped as actions:

* `apify`: [Apify][15] is a web scraping and web automation platform that enables you to build your
  own web crawlers and web scrapers.
* `bing_search`: Wrapper around the [Bing Web Search][16] API.
* `google_search`: Wrapper around the [Google Search API][17] from Langchain.
* `searx_search`: Wrapper around the [Searx][18] API. Alternative to Google/Bing Search.
* `google_serper`: Wrapper around the SerpApi [Google Search API][19]. It can be used to add answer
  boxes and knowledge graphs from Google Search.
* `openweather_query`: Wrapper around [OpenWeatherMap’s][20] API for retrieving weather information.
* `serp_api_query`: Wrapper around the [SerpAPI][21] API. It provides access to search engines and
  helps answer questions about current events.
* `wikipedia_query`: A wrapper around the [Wikipedia API][22]. It uses the MediaWiki API to retrieve
  information from Wikipedia.
* `wolfram_alpha_query`: A wrapper around the [Wolfram Alpha API][23]. It can be used to answer math
  and science questions.
* `zapier_nla_query`: Wrapper around the [Zapier NLA API][24]. It provides access to over 5k
  applications and 20k actions to automate your workflows.

### Chains as Actions[#][25]

> **⚠️ DEPRECATED**: Chain support is deprecated and will be removed in a future release. Please use
> [Runnable][26] instead. See the [Runnable as Action Guide][27] for examples.

You can register a Langchain chain as an action using the [LLMRails.register_action][28] method:

app.register_action(some_chain, name="some_chain")

When a chain is invoked as an action, the parameters of the action correspond to the input keys of
the chain. For the return value, if the output of the chain has a single key, the value will be
returned. If the chain has multiple output keys, the dictionary of output keys and their values is
returned. See the [LangChain Integration Guide][29] for more details.

### Custom Actions[#][30]

You can register any python function as a custom action, using the `action` decorator or with
`LLMRails(RailsConfig).register_action(action: callable, name: Optional[str])`.

from nemoguardrails.actions import action

@action()
async def some_action():
    # Do some work

    return "some_result"

By default, the name of the action is set to the name of the function. However, you can change it by
specifying a different name.

from nemoguardrails.actions import action

@action(name="some_action_name")
async def some_action():
    # Do some work

    return "some_result"

Actions can take any number of parameters. Since actions are invoked from Colang flows, the
parameters’ type is limited to *string*, *integer*, *float*, *boolean*, *list* and *dictionary*.

#### Special parameters[#][31]

The following parameters are special and are provided automatically by the NeMo Guardrails toolkit,
if they appear in the signature of an action:

* `events`: the history of events so far; the last one is the one triggering the action itself;
* `context`: the context data available to the action;
* `llm`: access to the LLM instance (BaseLLM from LangChain);
* `config`: the full `RailsConfig` instance.

These parameters are only meant to be used in advanced use cases.

## Action Parameters[#][32]

The following are the parameters that can be used in the actions:

───┬─────────────────────────┬───┬──────────────────────────────────────────────────────────────────
Par│Description              │Typ│Example                                                           
ame│                         │e  │                                                                  
ter│                         │   │                                                                  
s  │                         │   │                                                                  
───┼─────────────────────────┼───┼──────────────────────────────────────────────────────────────────
`ev│The history of events so │Lis│`[ {'type': 'UtteranceUserActionFinished', ...}, {'type':         
ent│far; the last one is the │t[d│'StartInternalSystemAction', 'action_name':                       
s` │one triggering the action│ict│'generate_user_intent', ...}, {'type':                            
   │itself.                  │]  │'InternalSystemActionFinished', 'action_name':                    
   │                         │   │'generate_user_intent', ...} ]`                                   
───┼─────────────────────────┼───┼──────────────────────────────────────────────────────────────────
`co│The context data         │dic│`{ 'last_user_message': ..., 'last_bot_message': ...,             
nte│available to the action. │t  │'retrieved_relevant_chunks': ... }`                               
xt`│                         │   │                                                                  
───┼─────────────────────────┼───┼──────────────────────────────────────────────────────────────────
`ll│Access to the LLM        │Bas│`OpenAI(model="gpt-3.5-turbo-instruct",...)`                      
m` │instance (BaseLLM from   │eLL│                                                                  
   │LangChain).              │M  │                                                                  
───┴─────────────────────────┴───┴──────────────────────────────────────────────────────────────────

[1]: #python-api
[2]: ../api/nemoguardrails.rails.llm.config.html#class-railsconfig
[3]: ../api/nemoguardrails.rails.llm.llmrails.html#class-llmrails
[4]: ../api/nemoguardrails.rails.llm.llmrails.html#method-llmrails-generate
[5]: ../api/nemoguardrails.rails.llm.llmrails.html#method-llmrails-generate-async
[6]: #basic-usage
[7]: #railsconfig
[8]: ../api/nemoguardrails.rails.llm.config.html#class-railsconfig
[9]: #message-generation
[10]: ../api/nemoguardrails.rails.llm.llmrails.html#method-llmrails-generate
[11]: ../api/nemoguardrails.rails.llm.llmrails.html#method-llmrails-generate-async
[12]: ../api/nemoguardrails.rails.llm.llmrails.html#method-llmrails-generate
[13]: #actions
[14]: #default-actions
[15]: https://python.langchain.com/en/latest/ecosystem/apify.html
[16]: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
[17]: https://python.langchain.com/en/latest/ecosystem/google_search.html
[18]: https://python.langchain.com/en/latest/ecosystem/searx.html
[19]: https://python.langchain.com/en/latest/ecosystem/google_serper.html
[20]: https://python.langchain.com/en/latest/modules/agents/tools/examples/openweathermap.html
[21]: https://python.langchain.com/en/latest/ecosystem/serpapi.html
[22]: https://python.langchain.com/en/latest/modules/agents/tools/examples/wikipedia.html
[23]: https://python.langchain.com/en/latest/ecosystem/wolfram_alpha.html
[24]: https://python.langchain.com/en/latest/modules/agents/tools/examples/zapier.html
[25]: #chains-as-actions
[26]: https://python.langchain.com/docs/expression_language/
[27]: langchain/runnable-as-action/README.html
[28]: ../api/nemoguardrails.rails.llm.llmrails.html#method-llmrails-register-action
[29]: langchain/langchain-integration.html
[30]: #custom-actions
[31]: #special-parameters
[32]: #action-parameters
