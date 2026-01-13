* Your first test

# Your first test[][1]

A Locust test is essentially just a Python program making requests to the system you want to test.
This makes it very flexible and particularly good at implementing complex user flows. But it can do
simple tests as well, so let’s start with that:

from locust import HttpUser, task

class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        self.client.get("/hello")
        self.client.get("/world")

This user will make an HTTP request to `/hello`, then to `/world`, and then repeat. For a full
explanation and a more realistic example see [Writing a locustfile][2].

Change `/hello` and `/world` to some actual paths on the website/service you want to test, put the
code in a file named `locustfile.py` in your current directory and then run `locust`:

$ locust
[2021-07-24 09:58:46,215] .../INFO/locust.main: Starting web interface at http://0.0.0.0:8089
[2021-07-24 09:58:46,285] .../INFO/locust.main: Starting Locust 2.42.6

## Locust’s web interface[][3]

Open [http://localhost:8089][4]

[_images/webui-splash-light.png]
Provide the host name of your server and try it out!

The following screenshots show what it might look like when running this test using 50 concurrent
users, with a ramp up rate of 1 user/s

[_images/webui-running-statistics-light.png]
Under the *Charts* tab you’ll find things like requests per second (RPS), response times and number
of running users:
[_images/bottlenecked-server-light.png]

Note

Interpreting performance test results is quite complex (and mostly out of scope for this manual),
but if your graphs start looking like this, the target service/system cannot handle the load and you
have found a bottleneck.

When we get to around 20 users, response times start increasing so fast that even though Locust is
still spawning more users, the number of requests per second is no longer increasing. The target
service is “overloaded” or “saturated”.

If your response times are *not* increasing then add even more users until you find the service’s
breaking point, or celebrate that your service is already performant enough for your expected load.

If you’re having trouble generating enough load to saturate your system, take a look at [Increasing
the request rate][5].

## Direct command line usage / headless[][6]

Using the Locust web UI is entirely optional. You can supply the load parameters on the command line
and get reports on the results in text form:

$ locust --headless --users 10 --spawn-rate 1 -H http://your-server.com
[2021-07-24 10:41:10,947] .../INFO/locust.main: No run time limit set, use CTRL+C to interrupt.
[2021-07-24 10:41:10,947] .../INFO/locust.main: Starting Locust 2.42.6
[2021-07-24 10:41:10,949] .../INFO/locust.runners: Ramping to 10 users using a 1.00 spawn rate
Name              # reqs      # fails  |     Avg     Min     Max  Median  |   req/s failures/s
----------------------------------------------------------------------------------------------
GET /hello             1     0(0.00%)  |     115     115     115     115  |    0.00    0.00
GET /world             1     0(0.00%)  |     119     119     119     119  |    0.00    0.00
----------------------------------------------------------------------------------------------
Aggregated             2     0(0.00%)  |     117     115     119     117  |    0.00    0.00
(...)
[2021-07-24 10:44:42,484] .../INFO/locust.runners: All users spawned: {"HelloWorldUser": 10} (10 tot
al users)
(...)

See [Running without the web UI][7] for more details.

## More options[][8]

To run Locust distributed across multiple Python processes or machines, you start a single Locust
master process with the `--master` command line parameter, and then any number of Locust worker
processes using the `--worker` command line parameter. See [Distributed load generation][9] for more
info.

To see all available options type: `locust --help` or check [Configuration][10].

## Next steps[][11]

Now, let’s have a more in-depth look at locustfiles and what they can do: [Writing a
locustfile][12].

[ Previous][13] [Next ][14]

© Copyright 2009-2025, Carl Byström, Jonatan Heyman, Lars Holmberg.

Built with [Sphinx][15] using a [theme][16] provided by [Read the Docs][17].

[1]: #your-first-test
[2]: writing-a-locustfile.html#writing-a-locustfile
[3]: #locust-s-web-interface
[4]: http://localhost:8089
[5]: increasing-request-rate.html#increaserr
[6]: #direct-command-line-usage-headless
[7]: running-without-web-ui.html#running-without-web-ui
[8]: #more-options
[9]: running-distributed.html#running-distributed
[10]: configuration.html#configuration
[11]: #next-steps
[12]: writing-a-locustfile.html#writing-a-locustfile
[13]: installation.html
[14]: writing-a-locustfile.html
[15]: https://www.sphinx-doc.org/
[16]: https://github.com/readthedocs/sphinx_rtd_theme
[17]: https://readthedocs.org
