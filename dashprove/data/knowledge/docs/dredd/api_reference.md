* »
* Hooks
* [ Edit on GitHub][1]

# Hooks[][2]

Dredd supports *hooks*, which are blocks of arbitrary code that run before or after each test step.
The concept is similar to XUnit’s `setUp` and `tearDown` functions, [Cucumber hooks][3], or [Git
hooks][4]. Hooks are usually used for:

* Loading database fixtures,
* cleaning up after test step(s),
* handling auth and sessions,
* passing data between transactions (saving state from responses),
* modifying a request generated from the API description,
* changing generated expectations,
* setting custom expectations,
* debugging by logging stuff.

## Getting started[][5]

Let’s have a description of a blog API, which allows to list all articles, and to publish a new one.

API BlueprintOpenAPI 2
FORMAT: 1A

# Blog API
## Articles [/articles]
### List articles [GET]

+ Response 200 (application/json; charset=utf-8)

        [
          {
            "id": 1,
            "title": "Creamy cucumber salad",
            "text": "Slice cucumbers…"
          }
        ]

### Publish an article [POST]

+ Request (application/json; charset=utf-8)

        {
          "title": "Crispy schnitzel",
          "text": "Prepare eggs…"
        }

+ Response 201 (application/json; charset=utf-8)

        {
          "id": 2,
          "title": "Crispy schnitzel",
          "text": "Prepare eggs…"
        }
swagger: "2.0"
info:
  title: "Blog API"
  version: "1.0"
consumes:
  - "application/json; charset=utf-8"
produces:
  - "application/json; charset=utf-8"
paths:
  "/articles":
    x-summary: "Articles"
    get:
      summary: "List articles"
      description: "Retrieve a list of all articles"
      responses:
        200:
          description: "Articles list"
          examples:
            "application/json; charset=utf-8":
              - id: 1
                title: "Creamy cucumber salad"
                text: "Slice cucumbers…"
    post:
      summary: "Publish an article"
      description: "Create and publish a new article"
      parameters:
        - name: "body"
          in: "body"
          schema:
            example:
              title: "Crispy schnitzel"
              text: "Prepare eggs…"
      responses:
        201:
          description: "New article"
          examples:
            "application/json; charset=utf-8":
              id: 2
              title: "Crispy schnitzel"
              text: "Prepare eggs…"

Now let’s say the real instance of the API has the POST request protected so it is not possible for
everyone to publish new articles. We do not want to hardcode secret tokens in our API description,
but we want to get Dredd to pass the auth. This is where the hooks can help.

### Writing hooks[][6]

Hooks are functions, which are registered to be ran for a specific test step (HTTP transaction) and
at a specific point in Dredd’s [execution life cycle][7]. Hook functions take one or more
[transaction objects][8], which they can modify. Let’s use hooks to add an [Authorization header][9]
to Dredd’s request.

Dredd supports [writing hooks in multiple programming languages][10], but we’ll go with JavaScript
hooks in this tutorial as they’re available out of the box.

API BlueprintOpenAPI 2

Let’s create a file called `hooks.js` with the following content:

const hooks = require('hooks');

hooks.before('Articles > Publish an article', (transaction) => {
  transaction.request.headers.Authorization = 'Basic: YWxhZGRpbjpvcGVuc2VzYW1l';
});

As you can see, we’re registering the hook function to be executed **before** the HTTP transaction
`Articles > Publish an article`. This path-like identifier is a [transaction name][11].

Let’s create a file called `hooks.js` with the following content:

const hooks = require('hooks');

hooks.before('Articles > Publish an article > 201 > application/json; charset=utf-8', (transaction) 
=> {
  transaction.request.headers.Authorization = 'Basic: YWxhZGRpbjpvcGVuc2VzYW1l';
});

As you can see, we’re registering the hook function to be executed **before** the HTTP transaction
`Articles > Publish an article > 201 > application/json`. This path-like identifier is a
[transaction name][12].

### Running Dredd with hooks[][13]

With the API instance running locally at `http://127.0.0.1:3000`, you can now run Dredd with hooks
using the [`--hookfiles`][14] option:

API BlueprintOpenAPI 2
dredd ./blog.apib http://127.0.0.1:3000 --hookfiles=./hooks.js
dredd ./blog.yaml http://127.0.0.1:3000 --hookfiles=./hooks.js

Now the tests should pass even if publishing new article requires auth.

## Supported languages[][15]

Dredd itself is written in JavaScript, so it supports [JavaScript hooks][16] out of the box. Running
hooks in other languages requires installing a dedicated *hooks handler*. Supported languages are:

* [JavaScript][17]
* [Go][18]
* [Perl][19]
* [PHP][20]
* [Python][21]
* [Ruby][22]
* [Rust][23]

Note

If you don’t see your favorite language, [it’s fairly easy to contribute support for it][24]! Join
the [Contributors Hall of Fame][25] where we praise those who added support for additional
languages.

(Especially if your language of choice is **Java**, there’s an eternal fame and glory waiting for
you - see [#875][26])

## Transaction names[][27]

Transaction names are path-like strings, which allow hook functions to address specific HTTP
transactions. They intuitively follow the structure of your API description document.

You can get a list of all transaction names available in your API description document by calling
Dredd with the [`--names`][28] option:

API BlueprintOpenAPI 2
$ dredd ./blog.apib http://127.0.0.1:3000 --names
info: Articles > List articles
skip: GET (200) /articles
info: Articles > Publish an article
skip: POST (201) /articles
complete: 0 passing, 0 failing, 0 errors, 2 skipped, 2 total
complete: Tests took 9ms

As you can see, the document `./blog.apib` contains two transactions, which you can address in hooks
as:

* `Articles > List articles`
* `Articles > Publish an article`
$ dredd ./blog.yaml http://127.0.0.1:3000 --names
info: Articles > List articles > 200 > application/json; charset=utf-8
skip: GET (200) /articles
info: Articles > Publish an article > 201 > application/json; charset=utf-8
skip: POST (201) /articles
complete: 0 passing, 0 failing, 0 errors, 2 skipped, 2 total
complete: Tests took 9ms

As you can see, the document `./blog.yaml` contains two transactions, which you can address in hooks
as:

* `Articles > List articles > 200 > application/json; charset=utf-8`
* `Articles > Publish an article > 201 > application/json; charset=utf-8`

Note

The transaction names and the [`--names`][29] workflow mostly do their job, but with [many
documented flaws][30]. A successor to transaction names is being designed in [#227][31]

## Types of hooks[][32]

Hooks get executed at specific points in Dredd’s [execution life cycle][33]. Available types of
hooks are:

* `beforeAll` called with all HTTP transactions before the whole test run
* `beforeEach` called before each HTTP transaction
* `before` called before a single HTTP transaction
* `beforeEachValidation` called before each HTTP transaction is validated
* `beforeValidation` called before a single HTTP transaction is validated
* `after` called after a single HTTP transaction
* `afterEach` called after each HTTP transaction
* `afterAll` called with all HTTP transactions after the whole test run

## Hooks inside Docker[][34]

As mentioned in [Supported languages][35], running hooks written in languages other than JavaScript
requires a dedicated hooks handler. Hooks handler is a separate process, which communicates with
Dredd over a TCP socket.

If you’re [running Dredd inside Docker][36], you may want to use a separate container for the hooks
handler and then run all your containers together as described in the [Docker Compose][37] section.

However, hooks were not originally designed with this scenario in mind. Dredd gets a name of (or
path to) the hooks handler in [`--language`][38] and then starts it as a child process. To work
around this, [fool Dredd with a dummy script][39] and set [`--hooks-worker-handler-host`][40]
together with [`--hooks-worker-handler-port`][41] to point Dredd’s TCP communication to the other
container.

Note

The issue described above is tracked in [#755][42].

[ Previous][43] [Next ][44]

Revision `3d4ae143`.

Built with [Sphinx][45] using a [theme][46] provided by [Read the Docs][47].

[1]: https://github.com/apiaryio/dredd/blob/master/docs/hooks/index.rst
[2]: #hooks
[3]: https://cucumber.io/docs/cucumber/api/#hooks
[4]: https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
[5]: #getting-started
[6]: #writing-hooks
[7]: ../how-it-works.html#execution-life-cycle
[8]: ../data-structures.html#transaction
[9]: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Authorization
[10]: #supported-languages
[11]: #transaction-names
[12]: #transaction-names
[13]: #running-dredd-with-hooks
[14]: ../usage-cli.html#cmdoption-hookfiles
[15]: #supported-languages
[16]: js.html#hooks-js
[17]: js.html
[18]: go.html
[19]: perl.html
[20]: php.html
[21]: python.html
[22]: ruby.html
[23]: rust.html
[24]: new-language.html#hooks-new-language
[25]: ../internals.html#maintainers
[26]: https://github.com/apiaryio/dredd/issues/875
[27]: #getting-transaction-names
[28]: ../usage-cli.html#cmdoption-names
[29]: ../usage-cli.html#cmdoption-names
[30]: https://github.com/apiaryio/dredd/labels/Epic%3A%20Transaction%20Names
[31]: https://github.com/apiaryio/dredd/issues/227
[32]: #types-of-hooks
[33]: ../how-it-works.html#execution-life-cycle
[34]: #hooks-inside-docker
[35]: #supported-languages
[36]: ../installation.html#docker
[37]: ../installation.html#docker-compose
[38]: ../usage-cli.html#cmdoption-language
[39]: https://github.com/apiaryio/dredd/issues/748#issuecomment-285355519
[40]: ../usage-cli.html#cmdoption-hooks-worker-handler-host
[41]: ../usage-cli.html#cmdoption-hooks-worker-handler-port
[42]: https://github.com/apiaryio/dredd/issues/755
[43]: ../usage-js.html
[44]: js.html
[45]: https://www.sphinx-doc.org/
[46]: https://github.com/readthedocs/sphinx_rtd_theme
[47]: https://readthedocs.org
