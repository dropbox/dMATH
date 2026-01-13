* Getting Started
On this page

# Getting Started with Pyre

Welcome to the wonderful world of static typing! This guide will get you from zero to a simple
project that is type checked with Pyre.

## Requirements[​][1]

To get started, you need [Python 3.8 or later][2] and [watchman][3] working on your system. On
*MacOS* you can get everything with [homebrew][4]:

`$ brew install python3 watchman
`

On *Ubuntu*, *Mint*, or *Debian*; use `apt-get` and [homebrew][5]:

`$ sudo apt-get install python3 python3-pip python3-venv watchman
`

We tested Pyre on *Ubuntu 18.04.5 LTS*, *CentOS 7*, as well as *OSX 10.11* and later.

## Setting up a Project[​][6]

We start by creating an empty project directory and setting up a virtual environment:

`$ mkdir my_project && cd my_project
$ python3 -m venv ~/.venvs/venv
$ source ~/.venvs/venv/bin/activate
(venv) $ pip install pyre-check
`

Next, we teach Pyre about our new project:

`(venv) $ pyre init
`

This command will set up a configuration for Pyre (`.pyre_configuration`) as well as watchman
(`.watchmanconfig`) in your project's directory. Accept the defaults for now – you can change them
later if necessary.

## Running Pyre[​][7]

We are now ready to run Pyre:

`(venv) $ echo "i: int = 'string'" > test.py
(venv) $ pyre
 ƛ Found 1 type error!
test.py:1:0 Incompatible variable type [9]: i is declared to have type `int` but is used as type `st
r`.
`

This first invocation will start a daemon listening for filesystem changes – type checking your
project incrementally as you make edits to the code. You will notice that subsequent invocations of
`pyre` will be faster than the first one.

## Introductory Video[​][8]

## Further Reading[​][9]

This page should contain all of the basic information you need to get started with type checking
your own project.

If you are new to the type system, the [introduction to types in Python][10] is recommended reading
to familiarize with the type system, gradual typing, and common type errors.

If you are looking for more options to configure your type checking experience, the
[configuration][11] page explores command line and configuration file settings.

[Edit this page][12]

[1]: #requirements
[2]: https://www.python.org/getit/
[3]: https://facebook.github.io/watchman/
[4]: https://brew.sh/
[5]: https://brew.sh/
[6]: #setting-up-a-project
[7]: #running-pyre
[8]: #introductory-video
[9]: #further-reading
[10]: /docs/types-in-python/
[11]: /docs/configuration/
[12]: https://github.com/facebook/pyre-check/tree/main/documentation/website/docs/getting_started.md
