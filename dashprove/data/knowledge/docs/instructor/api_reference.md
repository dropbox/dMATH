# Instructor Concepts[¶][1]

This section explains the core concepts and features of the Instructor library, organized by
category to help you find what you need.

## Core Concepts[¶][2]

These are the fundamental concepts you need to understand to use Instructor effectively:

* [Models][3] - Using Pydantic models to define output structures
* [Patching][4] - How Instructor patches LLM clients
* [from_provider][5] - Unified interface for creating clients across all providers
* [Migration Guide][6] - Migrating from older patterns to from_provider
* [Types][7] - Working with different data types in your models
* [Validation][8] - Validating LLM outputs against your models
* [Prompting][9] - Creating effective prompts for structured output extraction
* [Multimodal][10] - Working with Audio Files, Images and PDFs

## Data Handling and Structures[¶][11]

These concepts relate to defining and working with different data structures:

* [Fields][12] - Working with Pydantic fields and attributes
* [Lists and Arrays][13] - Handling lists and arrays in your models
* [TypedDicts][14] - Using TypedDict for flexible typing
* [Union Types][15] - Working with union types
* [Enums][16] - Using enumerated types in your models
* [Missing][17] - Handling missing or optional values
* [Alias][18] - Create field aliases

## Streaming Features[¶][19]

These features help you work with streaming responses:

* [Stream Partial][20] - Stream partially completed responses
* [Stream Iterable][21] - Stream collections of completed objects
* [Raw Response][22] - Access the raw LLM response

## Error Handling and Validation[¶][23]

These features help you ensure data quality:

* [Retrying][24] - Configure automatic retry behavior
* [Validators][25] - Define custom validation logic
* [Hooks][26] - Add callbacks for monitoring and debugging

## Performance Optimization[¶][27]

These features help you optimize performance:

* [Caching][28] - Cache responses to improve performance
* [Prompt Caching][29] - Cache prompts to reduce token usage
* [Usage Tokens][30] - Track token usage
* [Parallel Tools][31] - Run multiple tools in parallel

## Integration Features[¶][32]

These features help you integrate with other technologies:

* [FastAPI][33] - Integrate with FastAPI
* [Type Adapter][34] - Use TypeAdapter with Instructor
* [Templating][35] - Use templates for dynamic prompts
* [Distillation][36] - Optimize models for production

## Philosophy[¶][37]

* [Philosophy][38] - The guiding principles behind Instructor

## How These Concepts Work Together[¶][39]

Instructor is built around a few key ideas that work together:

1. **Define Structure with Pydantic**: Use Pydantic models to define exactly what data you want.
2. **Create Clients with from_provider**: Use the unified interface to create clients for any
   provider.
3. **Validate and Retry**: Automatically validate responses and retry if necessary.
4. **Process Streams**: Handle streaming responses for real-time updates.

### Typical Workflow[¶][40]

`sequenceDiagram
    participant User as Your Code
    participant Instructor
    participant LLM as LLM Provider

    User->>Instructor: Define Pydantic model
    User->>Instructor: Create client with from_provider
    User->>Instructor: Call create() with response_model
    Instructor->>LLM: Send structured request
    LLM->>Instructor: Return LLM response
    Instructor->>Instructor: Validate against model

    alt Validation Success
        Instructor->>User: Return validated Pydantic object
    else Validation Failure
        Instructor->>LLM: Retry with error context
        LLM->>Instructor: Return new response
        Instructor->>Instructor: Validate again
        Instructor->>User: Return validated object or error
    end`

## What to Read Next[¶][41]

* If you're new to Instructor, start with [Models][42] and [from_provider][43]
* If you're migrating from older patterns, see the [Migration Guide][44]
* If you're having validation issues, check out [Validators][45] and [Retrying][46]
* For streaming applications, read [Stream Partial][47] and [Stream Iterable][48]
* To optimize your application, look at [Caching][49] and [Usage Tokens][50]

For practical examples of these concepts, visit the [Cookbook][51] section.

See Also

* [Getting Started Guide][52] - Begin your journey with Instructor
* [Examples][53] - Practical implementations of these concepts
* [Integrations][54] - Connect with different LLM providers
Was this page helpful?
Thanks for your feedback!
Thanks for your feedback! Help us improve this page by using our [feedback form][55].

[1]: #instructor-concepts
[2]: #core-concepts
[3]: models/
[4]: patching/
[5]: from_provider/
[6]: migration/
[7]: types/
[8]: validation/
[9]: prompting/
[10]: multimodal/
[11]: #data-handling-and-structures
[12]: fields/
[13]: lists/
[14]: typeddicts/
[15]: unions/
[16]: enums/
[17]: maybe/
[18]: alias/
[19]: #streaming-features
[20]: partial/
[21]: iterable/
[22]: raw_response/
[23]: #error-handling-and-validation
[24]: retrying/
[25]: reask_validation/
[26]: hooks/
[27]: #performance-optimization
[28]: caching/
[29]: prompt_caching/
[30]: usage/
[31]: parallel/
[32]: #integration-features
[33]: fastapi/
[34]: typeadapter/
[35]: templating/
[36]: distillation/
[37]: #philosophy
[38]: philosophy/
[39]: #how-these-concepts-work-together
[40]: #typical-workflow
[41]: #what-to-read-next
[42]: models/
[43]: from_provider/
[44]: migration/
[45]: reask_validation/
[46]: retrying/
[47]: partial/
[48]: iterable/
[49]: caching/
[50]: usage/
[51]: ../examples/
[52]: ../getting-started/
[53]: ../examples/
[54]: ../integrations/
[55]: https://forms.gle/ijr9Zrcg2QWgKoWs7
