* [Using Flake8][1]
* Full Listing of Options and Their Descriptions
* [ View page source][2]

# Full Listing of Options and Their Descriptions[][3]

## Index of Options[][4]

* [`flake8 --version`][5]
* [`flake8 --help`][6]
* [`flake8 --verbose`][7]
* [`flake8 --quiet`][8]
* [`flake8 --color`][9]
* [`flake8 --count`][10]
* [`flake8 --exclude`][11]
* [`flake8 --extend-exclude`][12]
* [`flake8 --filename`][13]
* [`flake8 --stdin-display-name`][14]
* [`flake8 --format`][15]
* [`flake8 --hang-closing`][16]
* [`flake8 --ignore`][17]
* [`flake8 --extend-ignore`][18]
* [`flake8 --per-file-ignores`][19]
* [`flake8 --max-line-length`][20]
* [`flake8 --max-doc-length`][21]
* [`flake8 --indent-size`][22]
* [`flake8 --select`][23]
* [`flake8 --extend-select`][24]
* [`flake8 --disable-noqa`][25]
* [`flake8 --show-source`][26]
* [`flake8 --statistics`][27]
* [`flake8 --require-plugins`][28]
* [`flake8 --enable-extensions`][29]
* [`flake8 --exit-zero`][30]
* [`flake8 --jobs`][31]
* [`flake8 --output-file`][32]
* [`flake8 --tee`][33]
* [`flake8 --append-config`][34]
* [`flake8 --config`][35]
* [`flake8 --isolated`][36]
* [`flake8 --builtins`][37]
* [`flake8 --doctests`][38]
* [`flake8 --benchmark`][39]
* [`flake8 --bug-report`][40]
* [`flake8 --max-complexity`][41]

## Options and their Descriptions[][42]

* --version[][43]*
  [Go back to index][44]
  
  Show **Flake8**’s version as well as the versions of all plugins installed.
  
  Command-line usage:
  
  flake8 --version
  
  This **can not** be specified in config files.

* -h, --help[][45]*
  [Go back to index][46]
  
  Show a description of how to use **Flake8** and its options.
  
  Command-line usage:
  
  flake8 --help
  flake8 -h
  
  This **can not** be specified in config files.

* -v, --verbose[][47]*
  [Go back to index][48]
  
  Increase the verbosity of **Flake8**’s output. Each time you specify it, it will print more and
  more information.
  
  Command-line example:
  
  flake8 -vv
  
  This **can not** be specified in config files.

* -q, --quiet[][49]*
  [Go back to index][50]
  
  Decrease the verbosity of **Flake8**’s output. Each time you specify it, it will print less and
  less information.
  
  Command-line example:
  
  flake8 -q
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  quiet = 1

* --color[][51]*
  [Go back to index][52]
  
  Whether to use color in output. Defaults to `auto`.
  
  Possible options are `auto`, `always`, and `never`.
  
  This **can not** be specified in config files.
  
  When color is enabled, the following substitutions are enabled:
  
  * `%(bold)s`
  * `%(black)s`
  * `%(red)s`
  * `%(green)s`
  * `%(yellow)s`
  * `%(blue)s`
  * `%(magenta)s`
  * `%(cyan)s`
  * `%(white)s`
  * `%(reset)s`

* --count[][53]*
  [Go back to index][54]
  
  Print the total number of errors.
  
  Command-line example:
  
  flake8 --count dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  count = True

* --exclude=<patterns>[][55]*
  [Go back to index][56]
  
  Provide a comma-separated list of glob patterns to exclude from checks.
  
  This defaults to: `.svn,CVS,.bzr,.hg,.git,__pycache__,.tox,.nox,.eggs,*.egg`
  
  Example patterns:
  
  * `*.pyc` will match any file that ends with `.pyc`
  * `__pycache__` will match any path that has `__pycache__` in it
  * `lib/python` will look expand that using [`os.path.abspath()`][57] and look for matching paths
  
  Command-line example:
  
  flake8 --exclude=*.pyc dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  exclude =
      .tox,
      __pycache__

* --extend-exclude=<patterns>[][58]*
  [Go back to index][59]
  
  Added in version 3.8.0.
  
  Provide a comma-separated list of glob patterns to add to the list of excluded ones. Similar
  considerations as in [`--exclude`][60] apply here with regard to the value.
  
  The difference to the [`--exclude`][61] option is, that this option can be used to selectively add
  individual patterns without overriding the default list entirely.
  
  Command-line example:
  
  flake8 --extend-exclude=legacy/,vendor/ dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  extend-exclude =
      legacy/,
      vendor/
  extend-exclude = legacy/,vendor/

* --filename=<patterns>[][62]*
  [Go back to index][63]
  
  Provide a comma-separate list of glob patterns to include for checks.
  
  This defaults to: `*.py`
  
  Example patterns:
  
  * `*.py` will match any file that ends with `.py`
  * `__pycache__` will match any path that has `__pycache__` in it
  * `lib/python` will look expand that using [`os.path.abspath()`][64] and look for matching paths
  
  Command-line example:
  
  flake8 --filename=*.py dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  filename =
      example.py,
      another-example*.py

* --stdin-display-name=<display_name>[][65]*
  [Go back to index][66]
  
  Provide the name to use to report warnings and errors from code on stdin.
  
  Instead of reporting an error as something like:
  
  stdin:82:73 E501 line too long
  
  You can specify this option to have it report whatever value you want instead of stdin.
  
  This defaults to: `stdin`
  
  Command-line example:
  
  cat file.py | flake8 --stdin-display-name=file.py -
  
  This **can not** be specified in config files.

* --format=<format>[][67]*
  [Go back to index][68]
  
  Select the formatter used to display errors to the user.
  
  This defaults to: `default`
  
  By default, there are two formatters available:
  
  * default
  * pylint
  
  Other formatters can be installed. Refer to their documentation for the name to use to select
  them. Further, users can specify their own format string. The variables available are:
  
  * code
  * col
  * path
  * row
  * text
  
  The default formatter has a format string of:
  
  '%(path)s:%(row)d:%(col)d: %(code)s %(text)s'
  
  Command-line example:
  
  flake8 --format=pylint dir/
  flake8 --format='%(path)s::%(row)d,%(col)d::%(code)s::%(text)s' dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  format=pylint
  format=%(path)s::%(row)d,%(col)d::%(code)s::%(text)s

* --hang-closing[][69]*
  [Go back to index][70]
  
  Toggle whether pycodestyle should enforce matching the indentation of the opening bracket’s line.
  When you specify this, it will prefer that you hang the closing bracket rather than match the
  indentation.
  
  Command-line example:
  
  flake8 --hang-closing dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  hang_closing = True
  hang-closing = True

* --ignore=<errors>[][71]*
  [Go back to index][72]
  
  Specify a list of codes to ignore. The list is expected to be comma-separated, and does not need
  to specify an error code exactly. Since **Flake8** 3.0, this **can** be combined with
  [`--select`][73]. See [`--select`][74] for more information.
  
  For example, if you wish to only ignore `W234`, then you can specify that. But if you want to
  ignore all codes that start with `W23` you need only specify `W23` to ignore them. This also works
  for `W2` and `W` (for example).
  
  This defaults to: `E121,E123,E126,E226,E24,E704,W503,W504`
  
  Command-line example:
  
  flake8 --ignore=E121,E123 dir/
  flake8 --ignore=E24,E704 dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  ignore =
      E121,
      E123
  ignore = E121,E123

* --extend-ignore=<errors>[][75]*
  [Go back to index][76]
  
  Added in version 3.6.0.
  
  Specify a list of codes to add to the list of ignored ones. Similar considerations as in
  [`--ignore`][77] apply here with regard to the value.
  
  The difference to the [`--ignore`][78] option is, that this option can be used to selectively add
  individual codes without overriding the default list entirely.
  
  Command-line example:
  
  flake8 --extend-ignore=E4,E51,W234 dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  extend-ignore =
      E4,
      E51,
      W234
  extend-ignore = E4,E51,W234

* --per-file-ignores=<filename:errors>[ <filename:errors>][][79]*
  [Go back to index][80]
  
  Added in version 3.7.0.
  
  Specify a list of mappings of files and the codes that should be ignored for the entirety of the
  file. This allows for a project to have a default list of violations that should be ignored as
  well as file-specific violations for files that have not been made compliant with the project
  rules.
  
  This option supports syntax similar to [`--exclude`][81] such that glob patterns will also work
  here.
  
  This can be combined with both [`--ignore`][82] and [`--extend-ignore`][83] to achieve a full
  flexibility of style options.
  
  Command-line usage:
  
  flake8 --per-file-ignores='project/__init__.py:F401 setup.py:E121'
  flake8 --per-file-ignores='project/*/__init__.py:F401 setup.py:E121'
  
  This **can** be specified in config files.
  
  per-file-ignores =
      project/__init__.py:F401
      setup.py:E121
      other_project/*:W9

* --max-line-length=<n>[][84]*
  [Go back to index][85]
  
  Set the maximum length that any line (with some exceptions) may be.
  
  Exceptions include lines that are either strings or comments which are entirely URLs. For example:
  
  # https://some-super-long-domain-name.com/with/some/very/long/path
  
  url = '''\
      https://...
  '''
  
  This defaults to: `79`
  
  Command-line example:
  
  flake8 --max-line-length 99 dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  max-line-length = 79

* --max-doc-length=<n>[][86]*
  [Go back to index][87]
  
  Set the maximum length that a comment or docstring line may be.
  
  By default, there is no limit on documentation line length.
  
  Command-line example:
  
  flake8 --max-doc-length 99 dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  max-doc-length = 79

* --indent-size=<n>[][88]*
  [Go back to index][89]
  
  Set the number of spaces used for indentation.
  
  By default, 4.
  
  Command-line example:
  
  flake8 --indent-size 2 dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  indent-size = 2

* --select=<errors>[][90]*
  [Go back to index][91]
  
  **You usually do not need to specify this option as the default includes all installed plugin
  codes.**
  
  Specify the list of error codes you wish **Flake8** to report. Similarly to [`--ignore`][92]. You
  can specify a portion of an error code to get all that start with that string. For example, you
  can use `E`, `E4`, `E43`, and `E431`.
  
  Command-line example:
  
  flake8 --select=E431,E5,W,F dir/
  flake8 --select=E,W dir/
  
  This can also be combined with [`--ignore`][93]:
  
  flake8 --select=E --ignore=E432 dir/
  
  This will report all codes that start with `E`, but ignore `E432` specifically. This is more
  flexibly than the **Flake8** 2.x and 1.x used to be.
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  select =
      E431,
      W,
      F

* --extend-select=<errors>[][94]*
  [Go back to index][95]
  
  Added in version 4.0.0.
  
  **You usually do not need to specify this option as the default includes all installed plugin
  codes.**
  
  Specify a list of codes to add to the list of selected ones. Similar considerations as in
  [`--select`][96] apply here with regard to the value.
  
  The difference to the [`--select`][97] option is, that this option can be used to selectively add
  individual codes without overriding the default list entirely.
  
  Command-line example:
  
  flake8 --extend-select=E4,E51,W234 dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  extend-select =
      E4,
      E51,
      W234

* --disable-noqa[][98]*
  [Go back to index][99]
  
  Report all errors, even if it is on the same line as a `# NOQA` comment. `# NOQA` can be used to
  silence messages on specific lines. Sometimes, users will want to see what errors are being
  silenced without editing the file. This option allows you to see all the warnings, errors, etc.
  reported.
  
  Command-line example:
  
  flake8 --disable-noqa dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  disable_noqa = True
  disable-noqa = True

* --show-source[][100]*
  [Go back to index][101]
  
  Print the source code generating the error/warning in question.
  
  Command-line example:
  
  flake8 --show-source dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  show_source = True
  show-source = True

* --statistics[][102]*
  [Go back to index][103]
  
  Count the number of occurrences of each error/warning code and print a report.
  
  Command-line example:
  
  flake8 --statistics
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  statistics = True

* --require-plugins=<names>[][104]*
  [Go back to index][105]
  
  Require specific plugins to be installed before running.
  
  This option takes a list of distribution names (usually the name you would use when running `pip
  install`).
  
  Command-line example:
  
  flake8 --require-plugins=flake8-2020,flake8-typing-extensions dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  require-plugins =
      flake8-2020
      flake8-typing-extensions

* --enable-extensions=<errors>[][106]*
  [Go back to index][107]
  
  Enable [off-by-default][108] extensions.
  
  Plugins to **Flake8** have the option of registering themselves as off-by-default. These plugins
  will not be loaded unless enabled by this option.
  
  Command-line example:
  
  flake8 --enable-extensions=H111 dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  enable-extensions =
      H111,
      G123
  enable_extensions =
      H111,
      G123

* --exit-zero[][109]*
  [Go back to index][110]
  
  Force **Flake8** to use the exit status code 0 even if there are errors.
  
  By default **Flake8** will exit with a non-zero integer if there are errors.
  
  Command-line example:
  
  flake8 --exit-zero dir/
  
  This **can not** be specified in config files.

* --jobs=<n>[][111]*
  [Go back to index][112]
  
  Specify the number of subprocesses that **Flake8** will use to run checks in parallel.
  
  Note
  
  This option is ignored on platforms where `fork` is not a supported `multiprocessing` method.
  
  This defaults to: `auto`
  
  The default behaviour will use the number of CPUs on your machine as reported by
  [`multiprocessing.cpu_count()`][113].
  
  Command-line example:
  
  flake8 --jobs=8 dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  jobs = 8

* --output-file=<path>[][114]*
  [Go back to index][115]
  
  Redirect all output to the specified file.
  
  Command-line example:
  
  flake8 --output-file=output.txt dir/
  flake8 -vv --output-file=output.txt dir/

* --tee[][116]*
  [Go back to index][117]
  
  Also print output to stdout if output-file has been configured.
  
  Command-line example:
  
  flake8 --tee --output-file=output.txt dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  tee = True

* --append-config=<config>[][118]*
  [Go back to index][119]
  
  Added in version 3.6.0.
  
  Provide extra config files to parse in after and in addition to the files that **Flake8** found on
  its own. Since these files are the last ones read into the Configuration Parser, so it has the
  highest precedence if it provides an option specified in another config file.
  
  Command-line example:
  
  flake8 --append-config=my-extra-config.ini dir/
  
  This **can not** be specified in config files.

* --config=<config>[][120]*
  [Go back to index][121]
  
  Provide a path to a config file that will be the only config file read and used. This will cause
  **Flake8** to ignore all other config files that exist.
  
  Command-line example:
  
  flake8 --config=my-only-config.ini dir/
  
  This **can not** be specified in config files.

* --isolated[][122]*
  [Go back to index][123]
  
  Ignore any config files and use **Flake8** as if there were no config files found.
  
  Command-line example:
  
  flake8 --isolated dir/
  
  This **can not** be specified in config files.

* --builtins=<builtins>[][124]*
  [Go back to index][125]
  
  Provide a custom list of builtin functions, objects, names, etc.
  
  This allows you to let pyflakes know about builtins that it may not immediately recognize so it
  does not report warnings for using an undefined name.
  
  This is registered by the default PyFlakes plugin.
  
  Command-line example:
  
  flake8 --builtins=_,_LE,_LW dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  builtins =
      _,
      _LE,
      _LW

* --doctests[][126]*
  [Go back to index][127]
  
  Enable PyFlakes syntax checking of doctests in docstrings.
  
  This is registered by the default PyFlakes plugin.
  
  Command-line example:
  
  flake8 --doctests dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  doctests = True

* --benchmark[][128]*
  [Go back to index][129]
  
  Collect and print benchmarks for this run of **Flake8**. This aggregates the total number of:
  
  * tokens
  * physical lines
  * logical lines
  * files
  
  and the number of elapsed seconds.
  
  Command-line usage:
  
  flake8 --benchmark dir/
  
  This **can not** be specified in config files.

* --bug-report[][130]*
  [Go back to index][131]
  
  Generate information necessary to file a complete bug report for Flake8. This will pretty-print a
  JSON blob that should be copied and pasted into a bug report for Flake8.
  
  Command-line usage:
  
  flake8 --bug-report
  
  The output should look vaguely like:
  
  {
    "dependencies": [
      {
        "dependency": "setuptools",
        "version": "25.1.1"
      }
    ],
    "platform": {
      "python_implementation": "CPython",
      "python_version": "2.7.12",
      "system": "Darwin"
    },
    "plugins": [
      {
        "plugin": "mccabe",
        "version": "0.5.1"
      },
      {
        "plugin": "pycodestyle",
        "version": "2.0.0"
      },
      {
        "plugin": "pyflakes",
        "version": "1.2.3"
      }
    ],
    "version": "3.1.0.dev0"
  }
  
  This **can not** be specified in config files.

* --max-complexity=<n>[][132]*
  [Go back to index][133]
  
  Set the maximum allowed McCabe complexity value for a block of code.
  
  This option is provided by the `mccabe` dependency’s **Flake8** plugin.
  
  Command-line usage:
  
  flake8 --max-complexity 15 dir/
  
  This **can** be specified in config files.
  
  Example config file usage:
  
  max-complexity = 15
[ Previous][134] [Next ][135]

© Copyright 2016, Ian Stapleton Cordasco.

Built with [Sphinx][136] using a [theme][137] provided by [Read the Docs][138].

[1]: index.html
[2]: ../_sources/user/options.rst.txt
[3]: #full-listing-of-options-and-their-descriptions
[4]: #index-of-options
[5]: #cmdoption-flake8-version
[6]: #cmdoption-flake8-h
[7]: #cmdoption-flake8-v
[8]: #cmdoption-flake8-q
[9]: #cmdoption-flake8-color
[10]: #cmdoption-flake8-count
[11]: #cmdoption-flake8-exclude
[12]: #cmdoption-flake8-extend-exclude
[13]: #cmdoption-flake8-filename
[14]: #cmdoption-flake8-stdin-display-name
[15]: #cmdoption-flake8-format
[16]: #cmdoption-flake8-hang-closing
[17]: #cmdoption-flake8-ignore
[18]: #cmdoption-flake8-extend-ignore
[19]: #cmdoption-flake8-per-file-ignores
[20]: #cmdoption-flake8-max-line-length
[21]: #cmdoption-flake8-max-doc-length
[22]: #cmdoption-flake8-indent-size
[23]: #cmdoption-flake8-select
[24]: #cmdoption-flake8-extend-select
[25]: #cmdoption-flake8-disable-noqa
[26]: #cmdoption-flake8-show-source
[27]: #cmdoption-flake8-statistics
[28]: #cmdoption-flake8-require-plugins
[29]: #cmdoption-flake8-enable-extensions
[30]: #cmdoption-flake8-exit-zero
[31]: #cmdoption-flake8-jobs
[32]: #cmdoption-flake8-output-file
[33]: #cmdoption-flake8-tee
[34]: #cmdoption-flake8-append-config
[35]: #cmdoption-flake8-config
[36]: #cmdoption-flake8-isolated
[37]: #cmdoption-flake8-builtins
[38]: #cmdoption-flake8-doctests
[39]: #cmdoption-flake8-benchmark
[40]: #cmdoption-flake8-bug-report
[41]: #cmdoption-flake8-max-complexity
[42]: #options-and-their-descriptions
[43]: #cmdoption-flake8-version
[44]: #top
[45]: #cmdoption-flake8-h
[46]: #top
[47]: #cmdoption-flake8-v
[48]: #top
[49]: #cmdoption-flake8-q
[50]: #top
[51]: #cmdoption-flake8-color
[52]: #top
[53]: #cmdoption-flake8-count
[54]: #top
[55]: #cmdoption-flake8-exclude
[56]: #top
[57]: https://docs.python.org/3/library/os.path.html#os.path.abspath
[58]: #cmdoption-flake8-extend-exclude
[59]: #top
[60]: #cmdoption-flake8-exclude
[61]: #cmdoption-flake8-exclude
[62]: #cmdoption-flake8-filename
[63]: #top
[64]: https://docs.python.org/3/library/os.path.html#os.path.abspath
[65]: #cmdoption-flake8-stdin-display-name
[66]: #top
[67]: #cmdoption-flake8-format
[68]: #top
[69]: #cmdoption-flake8-hang-closing
[70]: #top
[71]: #cmdoption-flake8-ignore
[72]: #top
[73]: #cmdoption-flake8-select
[74]: #cmdoption-flake8-select
[75]: #cmdoption-flake8-extend-ignore
[76]: #top
[77]: #cmdoption-flake8-ignore
[78]: #cmdoption-flake8-ignore
[79]: #cmdoption-flake8-per-file-ignores
[80]: #top
[81]: #cmdoption-flake8-exclude
[82]: #cmdoption-flake8-ignore
[83]: #cmdoption-flake8-extend-ignore
[84]: #cmdoption-flake8-max-line-length
[85]: #top
[86]: #cmdoption-flake8-max-doc-length
[87]: #top
[88]: #cmdoption-flake8-indent-size
[89]: #top
[90]: #cmdoption-flake8-select
[91]: #top
[92]: #cmdoption-flake8-ignore
[93]: #cmdoption-flake8-ignore
[94]: #cmdoption-flake8-extend-select
[95]: #top
[96]: #cmdoption-flake8-select
[97]: #cmdoption-flake8-select
[98]: #cmdoption-flake8-disable-noqa
[99]: #top
[100]: #cmdoption-flake8-show-source
[101]: #top
[102]: #cmdoption-flake8-statistics
[103]: #top
[104]: #cmdoption-flake8-require-plugins
[105]: #top
[106]: #cmdoption-flake8-enable-extensions
[107]: #top
[108]: ../plugin-development/registering-plugins.html#off-by-default
[109]: #cmdoption-flake8-exit-zero
[110]: #top
[111]: #cmdoption-flake8-jobs
[112]: #top
[113]: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.cpu_count
[114]: #cmdoption-flake8-output-file
[115]: #top
[116]: #cmdoption-flake8-tee
[117]: #top
[118]: #cmdoption-flake8-append-config
[119]: #top
[120]: #cmdoption-flake8-config
[121]: #top
[122]: #cmdoption-flake8-isolated
[123]: #top
[124]: #cmdoption-flake8-builtins
[125]: #top
[126]: #cmdoption-flake8-doctests
[127]: #top
[128]: #cmdoption-flake8-benchmark
[129]: #top
[130]: #cmdoption-flake8-bug-report
[131]: #top
[132]: #cmdoption-flake8-max-complexity
[133]: #top
[134]: configuration.html
[135]: error-codes.html
[136]: https://www.sphinx-doc.org/
[137]: https://github.com/readthedocs/sphinx_rtd_theme
[138]: https://readthedocs.org
