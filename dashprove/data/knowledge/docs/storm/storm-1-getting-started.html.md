## 1. Obtain Storm

To be able to run Storm, you need to obtain it and run it on your system.
Currently, you can choose one of the following options:

* Build Storm [from source][1] on macOS or Linux
* Install Storm via a supported package manager
  
  * [Homebrew][2] on macOS
  * [AUR][3] on Arch Linux
* Use a [Docker container][4] on macOS, Linux or Windows
* Use a [virtual machine][5] on macOS, Linux or Windows

## 2. Prepare a model checking query

After you have obtained Storm, you need to make sure that your input model has
the right form. That is, on a fundamental level, you need to ensure that your
input model falls into one of the [model types][6] supported by Storm.

If your model indeed does, then the next thing is to have the model available in
an [input language][7] of Storm. If you don’t have such a model yet, you need to
first model the system you are interested in or transcribe it from a different
input format.

However, the input model is only “half” of the input you need to provide, the
other “half” being the property you want to verify. Please consult our [guide to
properties][8] for details on how to specify them.

An extensive list of example models and properties is available at the
[Quantitative Verification Benchmark Set][9].

## 3. Run Storm

Finally, if both the input model as well as the property are captured in an
appropriate format, then you are ready to run Storm!

Our [guide][10] illustrates how you can do so. Since the calls (and even the
binaries) you need to invoke depend on the input language for each of the input
languages, the guide shows how to run Storm depending on the input you have.

[1]: documentation/obtain-storm/build.html
[2]: documentation/obtain-storm/homebrew.html
[3]: https://aur.archlinux.org/packages/stormchecker
[4]: documentation/obtain-storm/docker.html
[5]: documentation/obtain-storm/vm.html
[6]: documentation/background/models.html
[7]: documentation/background/languages.html
[8]: documentation/background/properties.html
[9]: https://qcomp.org/benchmarks
[10]: documentation/usage/running-storm.html
