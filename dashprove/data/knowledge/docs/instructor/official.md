# Instructor: Top Multi-Language Library for Structured LLM Outputs[¶][1]

*Extract structured data from any LLM with type safety, validation, and automatic retries. Available
in Python, TypeScript, Go, Ruby, Elixir, and Rust.*

[[PyPI - Version]][2] [[License]][3] [[GitHub Repo stars]][4] [[Downloads]][5] [[Discord]][6]
[[Twitter Follow]][7]

> **Instructor for extraction, PydanticAI for agents.** Instructor shines when you need fast,
> schema-first extraction without extra agents. When your project needs quality gates, shareable
> runs, or built-in observability, try [PydanticAI][8]. PydanticAI is the official agent runtime
> from the Pydantic team: it adds typed tools, dataset replays, and production dashboards while
> keeping your existing Instructor models. Read the [PydanticAI docs][9] to see how to bring those
> capabilities into your stack.

## What is Instructor?[¶][10]

Instructor is the **most popular Python library** for extracting structured data from Large Language
Models (LLMs). With over **3 million monthly downloads, 11k stars, and 100+ contributors**, it's the
go-to solution for developers who need reliable, validated outputs from AI models.

Built on top of **Pydantic**, Instructor provides type-safe data extraction with automatic
validation, retries, and streaming support. Whether you're using OpenAI's GPT models, Anthropic's
Claude, Google's Gemini, **open source models with Ollama**, **DeepSeek**, or any of 15+ supported
providers, Instructor ensures your LLM outputs are always structured and validated.

## Key Features for LLM Data Extraction[¶][11]

* **Structured Outputs**: Define Pydantic models to specify exactly what data you want from your LLM
* **Automatic Retries**: Built-in retry logic when validation fails - no more manual error handling
* **Data Validation**: Leverage Pydantic's powerful validation to ensure response quality
* **Streaming Support**: Real-time processing of partial responses and lists
* **Multi-Provider**: Works with OpenAI, Anthropic, Google, Mistral, Cohere, Ollama, DeepSeek, and
  15+ LLM providers
* **Type Safety**: Full IDE support with proper type inference and autocompletion
* **Open Source Support**: Run any open source model locally with Ollama, llama-cpp-python, or vLLM

## Quick Start[¶][12]

Install Instructor and start extracting structured data in minutes:

pipuvpoetry
`pip install instructor
`
`uv add instructor
`
`poetry add instructor
`

### Extract Structured Data[¶][13]

Instructor's **`from_provider`** function provides a unified interface to work with any LLM
provider. Switch between OpenAI, Anthropic, Google, Ollama, DeepSeek, and 15+ providers with the
same code:

`import instructor
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    occupation: str


# Works with any provider - same interface everywhere
client = instructor.from_provider("openai/gpt-5-nano")
# Or: instructor.from_provider("anthropic/claude-3")
# Or: instructor.from_provider("google/gemini-pro")
# Or: instructor.from_provider("ollama/llama3")  # local

# Extract structured data from natural language
person = client.create(
    response_model=Person,
    messages=[
        {"role": "user", "content": "Extract: John is a 30-year-old software engineer"}
    ],
)
print(person)  # Person(name='John', age=30, occupation='software engineer')
`

The **`from_provider`** API supports both sync and async usage (`async_client=True`) and
automatically handles provider-specific configurations. [See all supported providers →][14]

## Complex Schemas & Validation[¶][15]

Instructor excels at extracting complex, nested data structures with custom validation rules. Here's
a concise example:

`import instructor
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Ticket(BaseModel):
    title: str = Field(..., min_length=5, max_length=100)
    priority: Priority
    estimated_hours: Optional[float] = Field(None, gt=0, le=100)

    @field_validator('estimated_hours')
    @classmethod
    def validate_hours(cls, v):
        if v is not None and v % 0.5 != 0:
            raise ValueError('Hours must be in 0.5 increments')
        return v

class CustomerSupport(BaseModel):
    customer_name: str
    tickets: List[Ticket] = Field(..., min_items=1)

client = instructor.from_provider("openai/gpt-4o")

support_case = client.create(
    response_model=CustomerSupport,
    messages=[{"role": "user", "content": "Extract support case details..."}],
    max_retries=3
)
`

**Key Features:** - Deep nesting with nested models and lists - Custom validation with Pydantic
validators - Automatic retries on validation failures - Type-safe extraction with full IDE support

[Learn more about validation and complex schemas →][16]

## Supported LLM Providers[¶][17]

Instructor works seamlessly with **15+ popular LLM providers**, giving you the flexibility to use
any model while maintaining consistent structured output handling. From OpenAI's GPT models to
**open source alternatives with Ollama**, **DeepSeek models**, and local inference, get validated
data extraction everywhere.

It stands out for its simplicity, transparency, and user-centric design, built on top of Pydantic.
Instructor helps you manage [validation context][18], retries with [Tenacity][19], and streaming
[Lists][20] and [Partial][21] responses.

[ Star the Repo][22] [ Cookbooks][23] [ Prompting Guide][24]

If you ever get stuck, you can always run `instructor docs` to open the documentation in your
browser. It even supports searching for specific topics.

`instructor docs [QUERY]
`

### Provider Examples[¶][25]

All providers use the same simple interface. Here are quick examples for the most popular providers:

OpenAIAnthropicGoogle GeminiOllama (Local)
`import instructor
from pydantic import BaseModel

class ExtractUser(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-5-nano")
res = client.create(
    response_model=ExtractUser,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)
`

[Full OpenAI docs →][26]

`import instructor
from pydantic import BaseModel

class ExtractUser(BaseModel):
    name: str
    age: int

client = instructor.from_provider("anthropic/claude-3-5-sonnet-20240620")
resp = client.create(
    response_model=ExtractUser,
    messages=[{"role": "user", "content": "Extract Jason is 25 years old."}],
)
`

[Full Anthropic docs →][27]

`import instructor
from pydantic import BaseModel

class ExtractUser(BaseModel):
    name: str
    age: int

client = instructor.from_provider("google/gemini-2.5-flash")
resp = client.create(
    response_model=ExtractUser,
    messages=[{"role": "user", "content": "Extract Jason is 25 years old."}],
)
`

[Full Google docs →][28]

`import instructor
from pydantic import BaseModel

class ExtractUser(BaseModel):
    name: str
    age: int

client = instructor.from_provider("ollama/llama3")
resp = client.create(
    response_model=ExtractUser,
    messages=[{"role": "user", "content": "Extract Jason is 25 years old."}],
)
`

[Full Ollama docs →][29]

[View all 15+ providers →][30]

## Citation[¶][31]

If you use Instructor in your research or project, please cite it using:

`@software{liu2024instructor,
  author = {Jason Liu and Contributors},
  title = {Instructor: A library for structured outputs from large language models},
  url = {https://github.com/instructor-ai/instructor},
  year = {2024},
  month = {3}
}
`

## Why use Instructor?[¶][32]

* **Simple API with Full Prompt Control**
  
  Instructor provides a straightforward API that gives you complete ownership and control over your
  prompts. This allows for fine-tuned customization and optimization of your LLM interactions.
  
  [ Explore Concepts][33]
* **Multi-Language Support**
  
  Simplify structured data extraction from LLMs with type hints and validation.
  
  [ Python][34] · [ TypeScript][35] · [ Ruby][36] · [ Go][37] · [ Elixir][38] · [ Rust][39]
* **Reasking and Validation**
  
  Automatically reask the model when validation fails, ensuring high-quality outputs. Leverage
  Pydantic's validation for robust error handling.
  
  [ Learn about Reasking][40]
* **Streaming Support**
  
  Stream partial results and iterables with ease, allowing for real-time processing and improved
  responsiveness in your applications.
  
  [ Learn about Streaming][41]
* **Powered by Type Hints**
  
  Leverage Pydantic for schema validation, prompting control, less code, and IDE integration.
  
  [ Learn more][42]
* **Simplified LLM Interactions**
  
  Support for [OpenAI][43], [Anthropic][44], [Google][45], [Vertex AI][46], [Mistral/Mixtral][47],
  [Ollama][48], [llama-cpp-python][49], [Cohere][50], [LiteLLM][51].
  
  [ See Hub][52]

### Using Hooks[¶][53]

Instructor's hooks system lets you intercept and handle events during LLM interactions. Use hooks
for logging, monitoring, or custom error handling:

`import instructor
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-4o-mini")

# Attach hooks for logging and error handling
client.on("completion:kwargs", lambda **kw: print("Called with:", kw))
client.on("completion:error", lambda e: print(f"Error: {e}"))

user_info = client.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "Extract: John is 20 years old"}],
)
`

[Learn more about hooks →][54]

## Type Inference & Advanced Methods[¶][55]

Instructor provides full type inference for better IDE support and type safety. The client includes
specialized methods for different use cases:

**Basic extraction:**

`import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-4o-mini")
user = client.create(response_model=User, messages=[...])  # Type: User
`

**Async support:**

`client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)
user = await client.create(...)  # Type: User
`

**Access original completion:**

`user, completion = client.create_with_completion(...)  # Returns tuple
`

**Stream partial objects:**

`for partial in client.create_partial(...):  # Type: Generator[User, None]
    print(partial)
`

**Stream multiple objects:**

`for user in client.create_iterable(...):  # Type: Generator[User, None]
    print(user)
`

All methods provide full type inference for better IDE autocomplete and type checking.

## Frequently Asked Questions[¶][56]

### What is Instructor?[¶][57]

Instructor is a Python library that extracts structured, validated data from Large Language Models
(LLMs). It uses Pydantic models to define output schemas and automatically handles validation,
retries, and error handling.

### Which LLM providers does Instructor support?[¶][58]

Instructor supports 15+ providers including OpenAI, Anthropic, Google Gemini, Mistral, Cohere,
Ollama, DeepSeek, and many more. See our [integrations page][59] for the complete list.

### Do I need to know Pydantic to use Instructor?[¶][60]

Basic Pydantic knowledge helps, but you can get started with simple models. Instructor works with
any Pydantic BaseModel, and you can learn advanced features as you need them.

### How does Instructor compare to other libraries?[¶][61]

Instructor focuses specifically on structured outputs with automatic validation and retries. Unlike
larger frameworks, Instructor does one thing very well: getting reliable, validated data from LLMs.

### Can I use Instructor with open source models?[¶][62]

Yes! Instructor works with Ollama, llama-cpp-python, and other local models. See our [Ollama
integration guide][63] to get started.

### Does Instructor work with async code?[¶][64]

Yes, Instructor fully supports async/await. Use `async_client=True` when creating your client, then
use `await client.create()`.

[View all FAQs →][65]

## Templating[¶][66]

Instructor supports templating with Jinja, which lets you create dynamic prompts. This is useful
when you want to fill in parts of a prompt with data. Here's a simple example:

`import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4o-mini")


class User(BaseModel):
    name: str
    age: int


# Create a completion using a Jinja template in the message content
response = client.create(
    messages=[
        {
            "role": "user",
            "content": """Extract the information from the
            following text: {{ data }}`""",
        },
    ],
    response_model=User,
    context={"data": "John Doe is thirty years old"},
)

print(response)
#> User(name='John Doe', age=30)
`

[Learn more about templating :octicons-arrow-right:][67]

## Validation[¶][68]

You can also use Pydantic to validate your outputs and get the llm to retry on failure. Check out
our docs on [retrying][69] and [validation context][70].

`import instructor
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator

# Create instructor client
client = instructor.from_provider("openai/gpt-4o-mini")


class QuestionAnswer(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(llm_validator("don't say objectionable things", client=client)),
    ]


try:
    qa = QuestionAnswer(
        question="What is the meaning of life?",
        answer="The meaning of life is to be evil and steal",
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for QuestionAnswer
    answer
      Assertion failed, The statement promotes objectionable behavior by encouraging evil and steali
ng. [type=assertion_error, input_value='The meaning of life is to be evil and steal', input_type=str
]
    """
`

## Contributing[¶][71]

If you want to help out, checkout some of the issues marked as `good-first-issue` or `help-wanted`.
Found [here][72]. They could be anything from code improvements, a guest blog post, or a new cook
book.

## License[¶][73]

This project is licensed under the terms of the MIT License.

Was this page helpful?
Thanks for your feedback!
Thanks for your feedback! Help us improve this page by using our [feedback form][74].

[1]: #instructor-top-multi-language-library-for-structured-llm-outputs
[2]: https://pypi.org/project/instructor/
[3]: https://github.com/instructor-ai/instructor/blob/main/LICENSE
[4]: https://github.com/instructor-ai/instructor
[5]: https://pypi.org/project/instructor/
[6]: https://discord.gg/bD9YE9JArw
[7]: https://twitter.com/jxnlco
[8]: https://ai.pydantic.dev/
[9]: https://ai.pydantic.dev/
[10]: #what-is-instructor
[11]: #key-features-for-llm-data-extraction
[12]: #quick-start
[13]: #extract-structured-data
[14]: integrations/
[15]: #complex-schemas-validation
[16]: concepts/reask_validation/
[17]: #supported-llm-providers
[18]: concepts/reask_validation/
[19]: concepts/retrying/
[20]: concepts/lists/
[21]: concepts/partial/
[22]: https://github.com/jxnl/instructor
[23]: examples/
[24]: prompting/
[25]: #provider-examples
[26]: integrations/openai/
[27]: integrations/anthropic/
[28]: integrations/google/
[29]: integrations/ollama/
[30]: integrations/
[31]: #citation
[32]: #why-use-instructor
[33]: concepts/models/
[34]: https://python.useinstructor.com
[35]: https://js.useinstructor.com
[36]: https://ruby.useinstructor.com
[37]: https://go.useinstructor.com
[38]: https://hex.pm/packages/instructor
[39]: https://rust.useinstructor.com
[40]: concepts/reask_validation/
[41]: concepts/partial/
[42]: https://docs.pydantic.dev/
[43]: integrations/openai/
[44]: integrations/anthropic/
[45]: integrations/google/
[46]: integrations/vertex/
[47]: integrations/together/
[48]: integrations/ollama/
[49]: integrations/llama-cpp-python/
[50]: integrations/cohere/
[51]: integrations/litellm/
[52]: integrations/
[53]: #using-hooks
[54]: concepts/hooks/
[55]: #type-inference-advanced-methods
[56]: #frequently-asked-questions
[57]: #what-is-instructor_1
[58]: #which-llm-providers-does-instructor-support
[59]: integrations/
[60]: #do-i-need-to-know-pydantic-to-use-instructor
[61]: #how-does-instructor-compare-to-other-libraries
[62]: #can-i-use-instructor-with-open-source-models
[63]: integrations/ollama/
[64]: #does-instructor-work-with-async-code
[65]: faq/
[66]: #templating
[67]: concepts/templating/
[68]: #validation
[69]: concepts/retrying/
[70]: concepts/reask_validation/
[71]: #contributing
[72]: https://github.com/jxnl/instructor/labels/good%20first%20issue
[73]: #license
[74]: https://forms.gle/ijr9Zrcg2QWgKoWs7
