* [Support Center][1]
* [Documentation][2]
* [Desktop editions][3]
* [Extending Burp][4]
* [Extensions][5]
* [Creating][6]

ProfessionalCommunity Edition

# Creating Burp extensions

* **Last updated: ** December 16, 2025
* **Read time: ** 3 Minutes

Burp extensions are flexible and powerful plugins that enable you to customize Burp Suite to suit
your workflow.

You can create your own extensions in Java using our Montoya API. This enables you to, for example:

* Add new security testing features.
* Modify traffic handling.
* Integrate external tools.
* Interact with internal Burp tools.
* Extend Burp's user interface.

## Exploring your idea

Before you begin development, refine your idea and gather inspiration:

* **Browse the [BApp store][7]** - Check the list of BApps. You may find one that already meets your
  needs.
* **Look at example extensions** - Explore example extensions in the [Montoya API examples GitHub
  repository][8]. These showcase different ways to extend Burp.
* **Join the conversation** - Connect with other extension developers on the [PortSwigger
  Discord][9] #extensions channel.
* **Review our acceptance criteria** - If you plan to submit your extension to the BApp store, check
  our [BApp store acceptance criteria][10] before you begin.

## Choosing the right extensibility option

Burp offers different customization options depending on your needs:

* **Bambdas** are best for customizing match-and-replace rules, table columns, or custom filters.
  Bambdas are small sections of Java-based code that run directly in Burp, making them easier to
  write as they don't require project setup or UI configuration. For more information, see [Creating
  scripts][11].
* **BChecks** are best for defining custom scan checks. BChecks use an easy-to-learn, purpose-built
  language to create tailored checks. They provide a way to extend Burp Scanner without requiring a
  full extension. For more information, see [Creating custom scan checks][12].
* **Extensions** are best for complex functionality. They offer greater flexibility but require
  additional setup. To help you get started, we provide a [starter project][13] for managing
  dependencies and development.

## Choosing a language

We strongly recommend writing Java-based extensions that use the Montoya API. This modern API gives
you full access to Burp's extensibility features. It's actively maintained, well-documented, and
designed to be easy to work with. All of our documentation and examples are written in Java using
the Montoya API.

You can also write extensions in Kotlin, which can be compiled into a `.jar` file and loaded in Burp
like a Java extension.

#### Note

You can also write extensions in Python (via Jython) or Ruby (via JRuby) using the [legacy Extender
API][14]. Please be aware that the Extender API is no longer actively maintained.

For examples that use the legacy Extender API, see [Extender API examples (Legacy)][15].

## Creating AI-powered extensions

The Montoya API enables you to integrate advanced AI features into your Burp Suite extensions. Your
extensions can now send prompts to a Large Language Model (LLM), allowing real-time input analysis
and intelligent responses.

#### More information

For more information, see [Creating AI extensions][16].

## Vibe coding extensions with AI

LLMs can help you create Burp extensions more quickly and easily. To support this, our extension
starter project includes a `CLAUDE.md` file. The file is created for use with Claude Code, but can
be used with other LLMs. It provides the LLM with essential context on the Montoya API and Burp
extension development.

#### Related pages

For setup instructions, including how to use the `CLAUDE.md` file, see:

* [Setting up your extension development environment using the starter project][17].
* [Setting up your extension development environment manually][18].

## Handling kettled HTTP/2 requests in extensions

When issuing new requests from your extension, you're free to send [kettled requests][19] using
HTTP/2 formatting. This enables you to develop extensions to test for [HTTP/2-exclusive
vulnerabilities][20].

However, it is not currently possible for extensions to modify kettled requests that were issued by
Burp. This is because the API only allows them to access the normalized, HTTP/1 style request
representation.

## Custom editor best practice

Make sure that any `ExtensionHttpRequestEditor` returned does not use an `HttpRequestEditor` as the
UI component when it registers an `HttpRequestEditorProvider`. This avoids a scenario where the
`HttpRequestEditor` is created within another `HttpRequestEditor`, potentially creating an infinite
loop of `HttpRequestEditor` components and causing Burp to crash.

For the same reason, avoid returning an `HttpResponseEditor` when registering an
`HttpResponseEditorProvider`.

#### In this section

* [Setting up your extension development environment][21].
* [Writing your first extension][22]
* [Extension tutorials][23]
* [Loading your extension in Burp][24].
* [Setting up remote debugging][25].
* [Submitting extensions to the BApp store][26].
* [Maintaining extensions on the BApp Store][27]
  
  .
* [BApp store acceptance criteria][28].
* [Extender API examples][29]

[1]: https://portswigger.net/support
[2]: /burp/documentation
[3]: /burp/documentation/desktop
[4]: /burp/documentation/desktop/extend-burp
[5]: /burp/documentation/desktop/extend-burp/extensions
[6]: /burp/documentation/desktop/extend-burp/extensions/creating
[7]: https://portswigger.net/bappstore
[8]: https://github.com/PortSwigger/burp-extensions-montoya-api-examples
[9]: https://discord.com/invite/portswigger
[10]: /burp/documentation/desktop/extend-burp/extensions/creating/bapp-store-acceptance-criteria
[11]: /burp/documentation/desktop/extend-burp/bambdas/creating
[12]: /burp/documentation/desktop/extend-burp/custom-scan-checks/creating
[13]: /burp/documentation/desktop/extend-burp/extensions/creating/set-up/starter-project
[14]: https://portswigger.net/burp/extender/api/
[15]: /burp/documentation/desktop/extend-burp/extensions/creating/extender-api-examples-legacy
[16]: /burp/documentation/desktop/extend-burp/extensions/creating/creating-ai-extensions
[17]: /burp/documentation/desktop/extend-burp/extensions/creating/set-up/starter-project
[18]: /burp/documentation/desktop/extend-burp/extensions/creating/set-up/manual-setup
[19]: https://portswigger.net/burp/documentation/desktop/http2#kettled-requests
[20]: /burp/documentation/desktop/http2/performing-http2-exclusive-attacks
[21]: /burp/documentation/desktop/extend-burp/extensions/creating/set-up
[22]: /burp/documentation/desktop/extend-burp/extensions/creating/first-extension
[23]: /burp/documentation/desktop/extend-burp/extensions/creating/tutorials
[24]: /burp/documentation/desktop/extend-burp/extensions/creating/loading-in-burp
[25]: /burp/documentation/desktop/extend-burp/extensions/creating/debugging
[26]: /burp/documentation/desktop/extend-burp/extensions/creating/bapp-store-submitting-extensions
[27]: /burp/documentation/desktop/extend-burp/extensions/creating/bapp-store-maintaining-extensions
[28]: /burp/documentation/desktop/extend-burp/extensions/creating/bapp-store-acceptance-criteria
[29]: /burp/documentation/desktop/extend-burp/extensions/creating/extender-api-examples-legacy
