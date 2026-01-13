[Docs][1]ReferenceTest Scripts

# Test Scripts

## Structure of test definitions

An Artillery test definition is composed of two main parts:

* `config` - which defines the runtime configuration for the entire test, such as the target URL,
  load phases, and protocol-specific settings
* `scenarios` - which define the behavior of virtual users (VUs) in the test, such as the sequence
  of HTTP requests they make

## YAML, TypeScript, or JavaScript

Artillery scripts can be written using YAML, TypeScript or JavaScript.

If you’re using YAML, the test definition is written in a YAML file with a `.yml` or `.yaml`
extension. The file should have top-level `config` and `scenarios` attributes.

`config:
  target: "http://localhost:3000"
  phases:
    - duration: 60
      arrivalRate: 10
scenarios:
  - flow:
      - get:
          url: "/api/users"`

If you’re using TypeScript or JavaScript, the test definition is written in a `.ts` or `.js` file.
The file should have two named exports: `config` and `scenarios`.

`export const config = {
  target: 'http://localhost:3000',
  phases: [
    {
      duration: 60,
      arrivalRate: 10
    }
  ]
};
 
export const scenarios = [
  {
    flow: [
      {
        get: {
          url: '/api/users',
        }
      }
    ]
  }
];`

If you’re writing Playwright-based tests, your scenarios will be written using Playwright’s API. For
example:

`import { Page } from '@playwright/test';
 
export const config = {
  target: 'https://www.artillery.io',
  engines: {
    playwright: {}
  }
};
 
export const scenarios = [{
  engine: 'playwright',
  testFunction: helloWorld
}];
 
async function helloWorld(page: Page) {
  await page.goto('https://www.artillery.io/');
  await page.click('text=Docs');
}
 
export function $rewriteMetricName(metricName: string, metricType: string) {
  if (metricName.includes('/checkout?promoid=')) {
    return 'browser.page.checkout';
  } else {
    return metricName;
  }
}`

See the [Playwright reference][2] for more details.

## `config` section

The `config` section usually defines the target (the hostname or IP address of the system under
test), the load progression, and protocol-specific settings, such as HTTP response timeouts or
Socket.io transport options. It may also be used to load and configure plugins and custom JS code.

### `target` - target service

`config.target` sets the endpoint of the system under test, such as a hostname, an IP address or a
URI.

The format of this field depends on the system you’re testing and the environment it runs in. For
example, for an HTTP-based application, it’s typically the protocol + hostname (e.g.
`http://myapp.staging.local`). For a WebSocket server, it’s usually the hostname (and optionally the
port) of the server (e.g. `ws://127.0.0.1`), and so on.

### `phases` - load phases

A load phase defines how Artillery generates new virtual users (VUs) in a specified time period. For
example, a typical performance test will have a gentle warm-up phase, followed by a ramp-up phase,
and finalizing with a maximum load for a duration of time.

`config.phases` is an array of phase definitions that Artillery goes through sequentially.

#### Load Phase types

Four kinds of phases are supported:

1. A phase with a duration and a constant **arrival rate** of a number of new VUs per second
2. A linear **ramp-up** phase where the number of new arrivals increases linearly over time
3. A phase that generates a fixed **count** of new arrivals over a period of time
4. A **pause** phase which generates no new VUs for a duration of time

#### Load Phase - additional options

* `maxVusers`: You can cap the total number of VUs in any phase with this option. Use this to
  restrict concurrency.
* `name`: You can give a name to a phase to make it easier to identify in CLI logs and Artillery
  Cloud dashboards.
* `duration`: You can specify the duration of a phase in seconds or in a human-readable format (see
  below).

The duration of an arrival phase determines only how long virtual users will be generated for. It is
**not** the same as the duration of a test run. How long a given test will run for depends on
several factors, such as complexity and length of user scenarios, server response time, and network
latency.

#### Using time units for `duration` and `pause`

[Added inv2.0.0-37][3]

The default unit for `duration` and `pause` is seconds, and Artillery converts everything to seconds
under the hood.

However, you can also provide any human-readable format from the [ms package ][4]. Here’s a few
examples:

───────────────────────┬───────────────────────
Duration/Pause         │Conversion (in seconds)
───────────────────────┼───────────────────────
0.5h                   │1800                   
───────────────────────┼───────────────────────
45m, 45 minutes, 45min │2700                   
───────────────────────┼───────────────────────
3.5h, 3.5 hours, 3.5hrs│12600                  
───────────────────────┴───────────────────────

This is especially useful for longer durations (e.g. soak tests) where seconds might not be the best
way to visualise time.

#### Load phase examples

Constant arrival rate

The following example generates 50 virtual users every second for 5 minutes:

YAMLTypeScript

### YAML

`config:
  target: 'https://staging.example.com'
  phases:
    - duration: '5m'
      arrivalRate: 50`

### TypeScript

`export const config = {
  target: 'https://staging.example.com',
  phases: [
    {
      duration: '5m',
      arrivalRate: 50
    }
  ]
};`

The following example generates 10 virtual users every second for 5 minutes, with no more than 50
concurrent virtual users at any given time:

YAMLTypeScript

### YAML

`config:
  target: 'https://staging.example.com'
  phases:
    - duration: '5m'
      arrivalRate: 10
      maxVusers: 50`

### TypeScript

`export const config = {
  target: 'https://staging.example.com',
  phases: [
    {
      duration: '5m',
      arrivalRate: 10,
      maxVusers: 50
    }
  ]
};`
Ramp up rate

The following example ramps up the arrival rate of virtual users from 10 to 50 over 2 minutes:

YAMLTypeScript

### YAML

`config:
  target: 'https://staging.example.com'
  phases:
    - duration: '2m'
      arrivalRate: 10
      rampTo: 50`

### TypeScript

`export const config = {
  target: 'https://staging.example.com',
  phases: [
    {
      duration: '2m',
      arrivalRate: 10,
      rampTo: 50
    }
  ]
};`
Fixed number of arrivals per second

The following example creates 20 virtual users in 60 seconds (one virtual user approximately every 3
seconds):

YAMLTypeScript

### YAML

`config:
  target: 'https://staging.example.com'
  phases:
    - duration: '1m'
      arrivalCount: 20`

### TypeScript

`export const config = {
  target: 'https://staging.example.com',
  phases: [
    {
      duration: '1m',
      arrivalCount: 20
    }
  ]
};`
A do-nothing `pause` phase

The following example does not send any virtual users for 60 seconds:

YAMLTypeScript

### YAML

`config:
  target: 'https://staging.example.com'
  phases:
    - pause: 60`

### TypeScript

`export const config = {
  target: 'https://staging.example.com',
  phases: [
    {
      pause: 60
    }
  ]
};`
Using time unit conversion

The following is now a valid config describing two phases: a ramp up phase lasting 30 minutes,
followed by a sustain phase of 3 hours.

YAMLTypeScript

### YAML

`phases:
  - duration: 30m
    arrivalRate: 1
    rampTo: 100
    name: ramp up
  - duration: 3h
    arrivalRate: 100
    name: sustain`

### TypeScript

`export const config = {
  phases: [
    {
      duration: '30m',
      arrivalRate: 1,
      rampTo: 100,
      name: 'ramp up'
    },
    {
      duration: '3h',
      arrivalRate: 100,
      name: 'sustain'
    }
  ]
};`

#### How do ramps work?

Think of the `rampTo` setting as a shortcut for manually writing out a sequence of arrival phases.
For example, let’s say you have the following load phase defined:

`phases:
  - duration: 100
    arrivalRate: 1
    rampTo: 50`

The above load phase is equivalent to the following:

`phases:
  - arrivalRate: 1
    duration: 2
  - arrivalRate: 2
    duration: 2
  - arrivalRate: 3
    duration: 2
  -
    #  ... etc ...
  - arrivalRate: 50
    duration: 2`

Partial arrival rates are rounded up (ie: 1.5 arrivals -> 2 arrivals), this may happen in some
scenarios.

### `environments` - config profiles

Typically, you may want to reuse a load testing script across multiple environments with minor
tweaks. For instance, you may want to run the same performance tests in development, staging, and
production. However, for each environment, you need to set a different `target` and modify the load
phases.

Instead of duplicating your test definition files for each environment, you can use the
`config.environments` setting. It allows you to specify the number of named environments that you
can define with environment-specific configuration.

A typical use-case is to define multiple targets with different load phase definitions for each of
those systems:

YAMLTypeScript

### YAML

`config:
  target: 'http://service1.acme.corp:3003'
  phases:
    - duration: 10
      arrivalRate: 1
  environments:
    production:
      target: 'http://service1.prod.acme.corp:44321'
      phases:
        - duration: 1200
          arrivalRate: 10
    local:
      target: 'http://127.0.0.1:3003'
      phases:
        - duration: 1200
          arrivalRate: 20`

### TypeScript

`export const config = {
  target: 'http://service1.acme.corp:3003',
  phases: [
    {
      duration: 10,
      arrivalRate: 1
    }
  ],
  environments: {
    production: {
      target: 'http://service1.prod.acme.corp:44321',
      phases: [
        {
          duration: 1200,
          arrivalRate: 10
        }
      ]
    },
    local: {
      target: 'http://127.0.0.1',
      phases: [
        {
          duration: 1200,
          arrivalRate: 20
        }
      ]
    }
  }
};`

When running your performance test, you can specify the environment on the command line using the
`-e` flag. For example, to execute the example test script defined above with the `staging`
configuration:

`artillery run -e staging my-script.yml`

#### The `$environment` variable

When running your tests in a specific environment, you can access the name of the current
environment using the `$environment` variable.

For example, you can print the name of the current environment from a scenario during test
execution:

YAMLTypeScript

### YAML

`config:
  environments:
    local:
      target: 'http://127.0.0.1:3003'
      phases:
        - duration: 120
          arrivalRate: 20
scenarios:
  - flow:
      - log: 'Current environment is set to: {{ $environment }}'`

### TypeScript

`export const config = {
  environments: {
    local: {
      target: 'http://127.0.0.1:3003',
      phases: [
        {
          duration: 120,
          arrivalRate: 20
        }
      ]
    }
  }
};
 
export const scenarios = [
  {
    flow: [
      {
        log: 'Current environment is set to: {{ $environment }}'
      }
    ]
  }
];`

If you run the test with `artillery run -e local my-script.yml`, Artillery will print “Current
environment is set to: local”.

### `plugins` - plugin config

This section can be used to configure Artillery plugins. Please see [plugins overview][5] for
details.

### `processor` - load custom code

Artillery can run custom code via “hooks” at various points in the test lifecycle. For example, you
can use custom code to generate dynamic payloads, run custom checks, or track custom metrics.

Custom code is loaded through the `config.processor` attribute. The value of `config.processor`
should be the path to one of:

* A CommonJS module with a `.js` extension
* An ESM module with a `.mjs` extension [Added inv2.0.7][6]
* A TypeScript module with a `.ts` extension. Supported for local and AWS Fargate runs only, and
  currently does not work with AWS Lambda. [Added inv2.0.4][7]

For example, to load a CommonJS module from `./my-functions.js`:

`config:
  target: 'https://my.app.dev'
  phases:
    - duration: 300
      arrivalRate: 1
  processor: './my-functions.js'
scenarios:
  -  # ... scenarios definitions here ...`

#### Function signatures

Hook functions may be [`async`][8] starting from Artillery [v2.0.7][9]. Async functions that throw
an error stop the execution of the current VU.

Callback-based hooks will receive a `next()` callback that must be called with no arguments for the
scenario to continue. Calling the `next()` callback with an error object will stop the execution of
the current VU.

#### Preventing bundling of TypeScript packages

[Added inv2.0.6][10]

Artillery bundles your TypeScript code into a single CommonJS module. Sometimes you may run into
issues with the bundling of some npm packages. If that happens, you can mark specific packages as
external to prevent them from being bundled.

For example, to mark `lodash` and `zod` as external:

`config:
  bundling:
    external: ['lodash', 'zod']`

If you mark a package as external, you will need to ensure that it is available in the environment
where you run your test. In the case of Fargate tests, make sure to include a `package.json` file
next to the test script with the dependencies, so Fargate will install the dependencies in the
workers. For example:

`{
  "dependencies": {
    "lodash": "^4.17.21",
    "zod": "^3.0.0"
  }
}`

### `payload` - loading data from CSV files

You can use a CSV file to provide dynamic data to test scripts. For example, you might have a list
of usernames and passwords that you want to use to test authentication in your API. Artillery allows
you to load, parse and map data in CSV files to variables which can be used inside virtual user
scenarios.

The main use-case for loading data from CSV files is for randomizing request payloads. If you
require determinism, this feature may not work as expected. An example of determinism is making sure
that each row is not used more than once during a test run, or using the data from each row in
order.

Artillery supports two ways of providing data from a CSV file to virtual users:

1. A row at a time, i.e. each VU gets data from just one row
2. All rows, i.e. each VU has access to all of the data

For example, you may have a file named `users.csv` with the following contents:

`testuser1,password1
testuser2,password2
testuser3,password3`

To access this information in a test definition, you can load the data from the CSV file using
`config.payload` setting:

YAMLTypeScript

### YAML

`config:
  payload:
    # path is relative to the location of the test script
    path: 'users.csv'
    fields:
      - 'username'
      - 'password'
scenarios:
  - flow:
      - post:
          url: '/auth'
          json:
            username: '{{ username }}'
            password: '{{ password }}'`

### TypeScript

`export const config = {
  payload: {
    path: 'users.csv',
    fields: ['username', 'password']
  }
};
 
export const scenarios = [
  {
    flow: [
      {
        post: {
          url: '/auth',
          json: {
            username: '{{ username }}',
            password: '{{ password }}'
          }
        }
      }
    ]
  }
];`

In this example, we tell Artillery to load `users.csv` file with the `path` setting and make the
variables `username` and `password` available in scenarios containing values from one of the rows in
the CSV file.

We can also make the entire dataset available to every VU, using `loadAll`, and loop through it in
our scenario:

YAMLTypeScript

### YAML

`config:
  payload:
    path: 'users.csv'
    fields:
      - 'username'
      - 'password'
    loadAll: true
    name: auth # refer to the data as "auth"
scenarios:
  - flow:
      - loop:
          - post:
              url: '/auth'
              json:
                username: '{{ $loopElement.username }}'
                password: '{{ $loopElement.password }}'
        over: auth`

### TypeScript

`export const config = {
  payload: {
    path: 'users.csv',
    fields: ['username', 'password'],
    loadAll: true,
    name: 'auth' // refer to the data as "auth"
  }
};
 
export const scenarios = [
  {
    flow: [
      {
        loop: {
          over: 'auth',
          flow: [
            {
              post: {
                url: '/auth',
                json: {
                  username: '{{ $loopElement.username }}',
                  password: '{{ $loopElement.password }}'
                }
              }
            }
          ]
        }
      }
    ]
  }
];`

It’s also possible to import multiple CSV files in a test definition by setting `payload` as an
array:

YAMLTypeScript

### YAML

`payload:
  - path: 'pets.csv'
    fields:
      - 'species'
      - 'name'
  - path: 'urls.csv'
    fields:
      - 'url'`

### TypeScript

`export const config = {
  payload: [
    {
      path: 'pets.csv',
      fields: ['species', 'name']
    },
    {
      path: 'urls.csv',
      fields: ['url']
    }
  ]
};`

You can also dynamically load different CSV files depending on the environment you set with the `-e`
flag by using the `$environment` variable when specifying the `path`:

`payload:
  - path: '{{ $environment }}-logins.csv'
    fields:
      - 'username'
      - 'password'`

An example for dynamically loading a payload file is to load a different set of usernames and
passwords to use with an authentication endpoint when running the same test in different
environments.

#### Payload file options

* `fields` - Names of variables to use for each column in the CSV file
* `order` (default: `random`) - Control how rows are selected from the CSV file for each new virtual
  user.
  
  * This option may be set to `sequence` to iterate through the rows in a sequence (looping around
    and starting from the beginning after reaching the last row). Note that this **will not** work
    as expected when running distributed tests, as each node will have its own copy of the CSV data.
* `skipHeader` (default: `false`) - Set to `true` to make Artillery skip the first row in the file
  (typically the header row).
* `delimiter` (default: `,`) - If the payload file uses a delimiter other than a comma, set this
  option to the delimiter character.
* `cast` (default: `true`) - By default, Artillery will convert fields to native types (e.g. numbers
  or booleans). To keep those fields as strings, set this option to `false`.
* `skipEmptyLines` (default: `true`) - By default, Artillery skips empty lines in the payload. Set
  to `false` to include empty lines.
* `loadAll` and `name` - set `loadAll` to `true` to provide all rows to each VU, and `name` to a
  variable name which will contain the data

#### Example

The following example loads a payload file called `users.csv`, skips the first row, and selects each
subsequent row sequentially:

YAMLTypeScript

### YAML

`config:
    payload:
      path: "users.csv"
      fields:
        - "username"
        - "password"
      order: sequence
      skipHeader: true
  scenarios:
    - # ... the rest of the script`

### TypeScript

`export const config = {
  payload: {
    path: 'users.csv',
    fields: ['username', 'password'],
    order: 'sequence',
    skipHeader: true
  }
};`

### `variables` - inline variables

Variables can be defined in the `config.variables` section and used in scenario definitions.

Variables work similarly to loading fields from a payload file. You can define multiple values for a
variable and access them randomly in your scenarios. For instance, the following example defined two
variables, `{{ id }}` and `{{ postcode }}`, with multiple values:

YAMLTypeScript

### YAML

`config:
  target: 'http://app01.local.dev'
  phases:
    - duration: 300
      arrivalRate: 25
  variables:
    postcode:
      - 'SE1'
      - 'EC1'
      - 'E8'
      - 'WH9'
    id:
      - '8731'
      - '9965'
      - '2806'`

### TypeScript

`export const config = {
  target: 'http://app01.local.dev',
  phases: [
    { duration: 300, arrivalRate: 25 }
  ],
  variables: {
    postcode: ['SE1', 'EC1', 'E8', 'WH9'],
    id: ['8731', '9965', '2806']
  }
}`

Variables defined in this block are **only** available in scenario definitions. They cannot be used
to template any values in the `config` section of your scripts. If you need to dynamically override
values in the `config` section, use environment variables in conjunction with `$env`.

### `tls` - self-signed certificates

This setting may be used to tell Artillery to accept self-signed TLS certificates:

YAMLTypeScript

### YAML

`config:
  tls:
    rejectUnauthorized: false`

### TypeScript

`export const config = {
  tls: {
    rejectUnauthorized: false
  }
};`

Accepting self-signed certificates may be a security risk

### `ensure` - SLO checks

Please see the guide for [`ensure` plugin][11].

### `defaults` - Default config

You can set default config for your scenario through this option, e.g. `think` options.

This option is not recommended and may be deprecated in the future.

Please use [config.http.defaults][12] for the HTTP engine defaults instead.

### `includeFiles` - explicitly bundling files with the test

When running a test on [AWS Lambda][13] or [AWS Fargate][14], Artillery will automatically detect
any custom JS modules (including their npm dependencies) and CSV files used with the
`config.payload` setting and bundle them into the test package that gets sent to the workers.

You may want to include other files that Artillery cannot automatically detect, such as a file that
is read with `fs.readFileSync` in a custom function. You can use the `config.includeFiles` to
include those files:

YAMLTypeScript

### YAML

`config:
  target: 'https://example.net'
  includeFiles:
    - foo.json
    - bar.xml`

### TypeScript

`export const config = {
  target: 'https://example.net',
  includeFiles: ['foo.json', 'bar.xml']
};`

### Using environment variables

Values can be set dynamically via environment variables which are available under `$env` template
variable. This functionality helps set different configuration values without modifying the test
definition and keeping secrets out of your source code.

For example, to set a default HTTP header for all requests via the `SERVICE_API_KEY` environment
variable, your test definition would look like this:

YAMLTypeScript

### YAML

`config:
  target: 'https://service.acme.corp'
  phases:
    - duration: 600
      arrivalRate: 10
scenarios:
  - flow:
      - get:
          url: '/'
          headers:
            x-api-key: '{{ $env.SERVICE_API_KEY }}'`

### TypeScript

`export const config = {
  target: 'https://service.acme.org',
  phases: [
    { duration: 600, arrivalRate: 10 }
  ]
};
 
export const scenarios = [{
  flow: [{
    get: {
      url: '/',
      headers: {
        'x-api-key': '{{ $env.SERVICE_API_KEY }}'
      }
    }
  }]
}];`

You can keep the API key out of the source code and provide it on the fly when executing the test
script:

`export SERVICE_API_KEY="012345-my-api-key"
artillery run my-test.yaml`

You can also set multiple environment variables from a file using the [`--env-file` flag][15].

Environment variables were formerly accessible via `$processEnvironment` instead of `$env`. The old
name is still available, but will be deprecated in a future release.

[Added inv2.0.0-33][16]

## `scenarios` section

The `scenarios` section contains definitions for one or more scenarios for the virtual users (VUs)
that Artillery will create. Each scenario is a series of steps representing a typical sequence of
requests or messages sent by a user of an application.

A scenario definition is an object which requires a `flow` attribute and may contain additional
optional attributes:

* `flow` (required) - An array of operations that a virtual user performs. For example, you can
  execute GET and POST requests for an HTTP-based application or emit events for a Socket.IO test.
* `name` (optional) - Assign a descriptive name to a scenario, which can be helpful in reporting.
* `weight` (optional) - Allows for the probability of a scenario being picked by a new virtual user
  to be “weighed” relative to other scenarios.

Each Artillery engine used during testing supports additional scenario attributes. Read the
documentation to learn what you can do in a scenario for each Artillery engine:

* [Testing HTTP][17]
* [Testing with Playwright][18]
* [Testing Socket.IO][19]
* [Testing WebSockets][20]

### `before` and `after` sections

The `before` and `after` are optional top level sections that can be used to run an arbitrary
scenario once per test definition, before or after the `scenarios` section has run. Any variable
captured during the `before` execution will be available to all virtual users and to the `after`
scenario. These sections can be useful to set up or tear down test data.

When running in [distributed mode][21], `before` and `after` hooks will be executed **once per
worker**.

The following example calls an authentication endpoint and captures an auth token before the virtual
users arrive. After the scenarios have run, the `after` section invalidates the token:

`config:
  target: 'http://app01.local.dev'
  phases:
    - duration: 300
      arrivalRate: 25
 
before:
  flow:
    - log: 'Get auth token'
    - post:
        url: '/auth'
        json:
          username: 'myUsername'
          password: 'myPassword'
        capture:
          - json: $.id_token
            as: token
scenarios:
  - flow:
      - get:
          url: '/data'
          headers:
            authorization: 'Bearer {{ token }}'
after:
  flow:
    - log: 'Invalidate token'
    - post:
        url: '/logout'
        json:
          token: '{{ token }}'`

#### All engines supported

[Added inv2.0.4][22]

The `before` and `after` sections support usage of any engine (custom or built-in). You must make
sure to specify the engine desired in both `config` and the `before`/`after` section. If the engine
is not specified, the default engine (`http`) will be used.

For example, to use the `playwright` engine in the `before` section, you would specify the
following:

`config:
  ...
  engines:
    playwright: {}
  processor: ./processor.js
 
before:
  engine: playwright
  flowFunction: someActionFunction
 
scenarios:
  - engine: playwright
    flowFunction: yourFlowFunction`

### Scenario weights

Weights allow you to specify that some scenarios should be picked more often than others. If you
have three scenarios with weights `1`, `2`, and `5`, the scenario with the weight of `2` is twice as
likely to be picked as the one with a weight of `1`, and 2.5 times less likely than the one with a
weight of `5`. Or in terms of probabilities:

* scenario 1: 1/8 = 12.5% probability of being picked
* scenario 2: 2/8 = 25% probability of being picked
* scenario 3: 5/8 = 62.5% probability of being picked

Scenario weights are optional and set to `1` by default, meaning each scenario has the same
probability of getting picked.

#### Example of weight usage

`scenarios:
  # Approximately 60% of all VUs will run this scenario.
  - name: '/common route'
    weight: 6
    flow:
      - get:
          url: '/common'
 
  # Approximately 30% of all VUs will run this scenario.
  - name: '/average route'
    weight: 3
    flow:
      - get:
          url: '/average'
 
  # Approximately 10% of all VUs will run this scenario.
  - name: '/rare route'
    weight: 1
    flow:
      - get:
          url: '/rare'`

#### Running a single weighted scenario

[Added inv2.0.0-38][23]

You can use the flag `--scenario-name` to run a specific scenario, allowing you to reuse weighted
scenarios as individual scenarios. For example, to run the scenario named `/rare route` from the
example above:

`  artillery run --scenario-name "/rare route" my-test.yaml`

## Default variables

Artillery sets a number of template variables for each test run which are available in all test
scripts.

### Test-level variables

Test-level variables are available anywhere in the test script, i.e. in both the `config` and the
`scenarios` sections.

* `$env` - environment variables including those set through the `--env-file` flag
* `$testId` - unique ID of the current test run [Added inv2.0.6][24]
* `$environment` - the value of the environment flag (`-e` or `--environment`)
* `$dirname` - the directory of the test config (config or scenario file) [Added inv2.0.12][25]
* `target` - the value of `config.target` (or the `--target` flag)

### Scenario-level variables

Scenario-level variables are available only in the `scenarios` section of the test script.

* `$uuid` - unique ID of the virtual user
Last updated on November 18, 2025
[Examples][26][Artillery CLI][27]

[1]: /docs
[2]: /docs/reference/engines/playwright
[3]: https://github.com/artilleryio/artillery/releases/tag/v2.0.0-37
[4]: https://www.npmjs.com/package/ms
[5]: /docs/reference/extensions
[6]: https://github.com/artilleryio/artillery/releases/tag/artillery-2.0.7
[7]: https://github.com/artilleryio/artillery/releases/tag/artillery-2.0.4
[8]: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function
[9]: https://github.com/artilleryio/artillery/releases/tag/artillery-2.0.7
[10]: https://github.com/artilleryio/artillery/releases/tag/artillery-2.0.6
[11]: /docs/reference/extensions/ensure
[12]: ./engines/http#default-configuration
[13]: ../load-testing-at-scale/aws-lambda
[14]: ../load-testing-at-scale/aws-fargate
[15]: /docs/reference/cli/run
[16]: https://github.com/artilleryio/artillery/releases/tag/artillery-2.0.0-33
[17]: /docs/reference/engines/http
[18]: /docs/reference/engines/playwright
[19]: /docs/reference/engines/socketio
[20]: /docs/reference/engines/websocket
[21]: /docs/load-testing-at-scale
[22]: https://github.com/artilleryio/artillery/releases/tag/artillery-2.0.4
[23]: https://github.com/artilleryio/artillery/releases/tag/v2.0.0-38
[24]: https://github.com/artilleryio/artillery/releases/tag/artillery-2.0.6
[25]: https://github.com/artilleryio/artillery/releases/tag/artillery-2.0.12
[26]: /docs/get-started/examples
[27]: /docs/reference/cli
