* »
* Quickstart
* [ Edit on GitHub][1]

# Quickstart[][2]

In following tutorial you can quickly learn how to test a simple HTTP API application with Dredd.
The tested application will be very simple backend written in [Express.js][3].

## Install Dredd[][4]

$ npm install -g dredd

If you’re not familiar with the Node.js ecosystem or you bump into any issues, follow the
[installation guide][5].

## Document Your API[][6]

First, let’s design the API we are about to build and test. That means you will need to create an
API description file, which will document how your API should look like. Dredd supports two formats
of API description documents:

* [API Blueprint][7]
* [OpenAPI 2][8] (formerly known as Swagger)
API BlueprintOpenAPI 2

If you choose API Blueprint, create a file called `api-description.apib` in the root of your project
and save it with following content:

FORMAT: 1A

# GET /
+ Response 200 (application/json; charset=utf-8)

      {"message": "Hello World!"}

If you choose OpenAPI 2, create a file called `api-description.yml`:

swagger: '2.0'
info:
  version: '1.0'
  title: Example API
  license:
    name: MIT
host: www.example.com
basePath: /
schemes:
  - http
paths:
  /:
    get:
      produces:
        - application/json; charset=utf-8
      responses:
        '200':
          description: ''
          schema:
            type: object
            properties:
              message:
                type: string
            required:
              - message

## Implement Your API[][9]

As we mentioned in the beginning, we’ll use [Express.js][10] to implement the API. Install the
framework by `npm`:

$ npm init
$ npm install express --save

Now let’s code the thing! Create a file called `app.js` with following contents:

var app = require('express')();

app.get('/', function(req, res) {
  res.json({message: 'Hello World!'});
})

app.listen(3000);

## Test Your API[][11]

At this moment, the implementation is ready to be tested. Let’s run the server as a background
process and let’s test it:

$ node app.js &

Finally, let Dredd validate whether your freshly implemented API complies with the description you
have:

API BlueprintOpenAPI 2
$ dredd api-description.apib http://127.0.0.1:3000
$ dredd api-description.yml http://127.0.0.1:3000

## Configure Dredd[][12]

Dredd can be configured by [many CLI options][13]. It’s recommended to save your Dredd configuration
alongside your project, so it’s easier to repeatedly execute always the same test run. Use
interactive configuration wizard to create `dredd.yml` file in the root of your project:

$ dredd init
? Location of the API description document: api-description.apib
? Command to start API backend server e.g. (bundle exec rails server)
? URL of tested API endpoint: http://127.0.0.1:3000
? Programming language of hooks:
❯ nodejs
  python
  ruby
  ...
? Dredd is best served with Continuous Integration. Create CircleCI config for Dredd? Yes

Now you can start test run just by typing `dredd`!

$ dredd

## Use Hooks[][14]

Dredd’s [hooks][15] enable you to write some glue code in your favorite language to support enhanced
scenarios in your API tests. Read the documentation about hooks to learn more on how to write them.
Choose your language and install corresponding hooks handler library.

## Advanced Examples[][16]

For more complex example applications, please refer to:

* [Express.js][17]
* [Laravel][18]
* [Laravel & OpenAPI 3][19]
* [Ruby on Rails][20]
[ Previous][21] [Next ][22]

Revision `3d4ae143`.

Built with [Sphinx][23] using a [theme][24] provided by [Read the Docs][25].

[1]: https://github.com/apiaryio/dredd/blob/master/docs/quickstart.rst
[2]: #quickstart
[3]: http://expressjs.com/starter/hello-world.html
[4]: #install-dredd
[5]: installation.html#installation
[6]: #document-your-api
[7]: https://apiblueprint.org
[8]: https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md
[9]: #implement-your-api
[10]: http://expressjs.com/starter/hello-world.html
[11]: #test-your-api
[12]: #configure-dredd
[13]: usage-cli.html#usage-cli
[14]: #use-hooks
[15]: hooks/index.html#hooks
[16]: #advanced-examples
[17]: https://github.com/apiaryio/dredd-example
[18]: https://github.com/ddelnano/dredd-hooks-php/wiki/Laravel-Example
[19]: https://github.com/AndyWendt/laravel-dredd-openapi-v3
[20]: https://gitlab.com/theodorton/dredd-test-rails/
[21]: installation.html
[22]: how-it-works.html
[23]: https://www.sphinx-doc.org/
[24]: https://github.com/readthedocs/sphinx_rtd_theme
[25]: https://readthedocs.org
