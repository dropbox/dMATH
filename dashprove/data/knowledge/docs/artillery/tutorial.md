[Docs][1][Getting Started][2]Run Your First Test

# Run Your First Artillery Test

We are going to write & run an Artillery test script for a simple API.

The API is a service that returns ASCII pictures of things, and it runs on
[http://asciiart.artillery.io:8080 ][3].

Let’s look at an example response from the API:

curl

### curl

`curl http://asciiart.artillery.io:8080/dino`
`                       _. - ~ ~ ~ - .
   ..       __.    .-~               ~-.
   ((\     /   `}.~                     `.
    \\\   {     }               /     \   \
(\   \\~~^      }              |       }   \
 \`.-~ -@~      }  ,-.         |       )    \
 (___     ) _}   (    :        |    / /      `.
  `----._-~.     _\ \ |_       \   / /- _      `.
         ~~----~~  \ \| ~~--~~~(  + /     ~-.     ~- _
                   /  /         \  \          ~- . _ _~_-_.
                __/  /          _\  )
              .<___.'         .<___/
`

## Create an Artillery test script

Here’s the Artillery test script we will use to load test our API in its entirety. We define a
simple but realistic load test, in which virtual users created by Artillery will perform multiple
actions.

We’ll write this test script using Artillery’s YAML-based DSL (domain-specific language), but they
can also be written in JavaScript or TypeScript.

We’ll walk through it step-by-step next to understand what’s going on.

`config:
  # This is a test server run by team Artillery
  # It's designed to be highly scalable
  target: http://asciiart.artillery.io:8080
  phases:
    - duration: 60
      arrivalRate: 1
      rampTo: 5
      name: Warm up phase
    - duration: 60
      arrivalRate: 5
      rampTo: 10
      name: Ramp up load
    - duration: 30
      arrivalRate: 10
      rampTo: 30
      name: Spike phase
  # Load & configure a couple of useful plugins
  # https://artillery.io/docs/reference/extensions
  plugins:
    ensure:
      thresholds:
        - http.response_time.p99: 100
        - http.response_time.p95: 75
    apdex:
      threshold: 100
    metrics-by-endpoint: {}
scenarios:
  - flow:
      - loop:
          - get:
              url: '/dino'
          - get:
              url: '/pony'
          - get:
              url: '/armadillo'
        count: 100`

Save this script as `asciiart-load-test.yml`.

### Test script walkthrough

Artillery test scripts have two parts: `config` and `scenarios`:

* `config` is what defines **how** our load test will run, e.g. the URL of the system we’re testing,
  how much load will be generated, any plugins we want to use, and so on.
* `scenarios` is where we define what the virtual users created by Artillery will do. A scenario is
  usually a sequence of steps that describes a user session in the app.

This is how it all fits together in our script:

### Set the target

We set the target with:

`target: 'http://asciiart.artillery.io:8080'`

This means that all requests will use that base URL by default.

### Define load phases

We describe the load phases in the next section. Load phases tell Artillery how many virtual users
to create, and describe the shape of the load we want.

`phases:
  - duration: 60
    arrivalRate: 5
    rampTo: 10
    name: Warm up the API
  - duration: 60
    arrivalRate: 10
    rampTo: 50
    name: Ramp up to peak load
  - duration: 300
    arrivalRate: 50
    name: Sustained peak load`

In this test we’ve defined three distict phases:

1. `Warm up the API` - this phase will run for 60 seconds. Artillery will start by creating 5 new
   virtual users per second, and gradually ramp up to 10 new virtual users per second by the end of
   the phase.
2. `Ramp up to peak load` - this phase will also last for 60 seconds. Artillery will continue
   ramping up load from 10 to 50 virtual users per second.
3. `Sustained peak load` - this phase will run for 300 seconds. Artillery will create 50 new virtual
   users every second during this phase.

These phases will give us what’s commonly known as a “spike test”, where load on our API spikes to a
high-level for a short amount of time. In the real world this may be the result of something like a
newsletter going out, a flash sale, or an expected daily peak in usage of our app.

#### Plugins

Our script also loads a couple of built-in plugins to produce detailed metrics for every URL in the
test, and to set up automatic success checks for the results of the test.

### Define scenarios

`scenarios:
  - name: Get 3 animal pictures
    flow:
      -loop:
        - get:
            url: '/dino'
        - get:
            url: '/pony'
        - get:
            url: '/armadillo'
      count: 100`

Our `scenarios` section defines one scenario. Each virtual user created in this test will run this
scenario.

The scenario contains three steps, each of which is a `HTTP GET` request to a different endpoint. We
use the `loop` action to repeat those 3 requests 100 times.

Every virtual user running the scenario is completely independent of other virtual users running the
same scenario, just like users and API clients in the real world. No memory, network connections,
cookies, or other state is shared between virtual users.

## Run the load test

We can run the script with Artillery like this:

`artillery run asciiart-load-test.yml`

Artillery will run the test, launching new virtual users as defined by the `config.phases` spec. As
virtual users run their scenario and collect performance metrics, Artillery will print a report
every 10 seconds with a summary of collected metrics for that time period.

The output will look similar to this, with reports describing the number of virtual users created,
HTTP response codes, and response times from API endpoints we’re testing:

`Test run id: tjjxc_6wygz7zab57pffz45gcrj3wwwn4cf_dbnt
Phase started: Warm up the API (index: 0, duration: 60s) 13:00:00(+0100)

--------------------------------------
Metrics for period to: 13:00:10(+0100) (width: 9.9s)
--------------------------------------

http.codes.200: ................................................................ 2100
http.downloaded_bytes: ......................................................... 1407350
http.request_rate: ............................................................. 200/sec
http.requests: ................................................................. 30
http.response_time:
  min: ......................................................................... 13
  max: ......................................................................... 15
  mean: ........................................................................ 14.1
  median: ...................................................................... 13.9
  p95: ......................................................................... 13.9
  p99: ......................................................................... 13.9
http.responses: ................................................................ 210
vusers.completed: .............................................................. 700
vusers.created: ................................................................ 700
vusers.created_by_name.0: ...................................................... 700
vusers.failed: ................................................................. 0
vusers.session_length:
  min: ......................................................................... 177
  max: ......................................................................... 177
  mean: ........................................................................ 176.9
  median: ...................................................................... 175.9
  p95: ......................................................................... 175.9
  p99: ......................................................................... 175.9`

Congrats! You have just run your first load test with Artillery.

Last updated on December 16, 2025
[Set up Artillery CLI][4][Learn Core Concepts][5]

[1]: /docs
[2]: /docs/get-started/load-testing
[3]: http://asciiart.artillery.io:8080
[4]: /docs/get-started/get-artillery
[5]: /docs/get-started/core-concepts
