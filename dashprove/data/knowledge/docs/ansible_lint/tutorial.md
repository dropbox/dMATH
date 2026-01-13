# Using[¶][1]

## Using commands[¶][2]

After you install Ansible-lint, run `ansible-lint --help` to display available commands and their
options.

`$ ansible-lint --help
usage: ansible-lint [-h] [-P | -L | -T]
                    [-f {brief,full,md,json,codeclimate,quiet,pep8,sarif}]
                    [--sarif-file SARIF_FILE] [-q]
                    [--profile {min,basic,moderate,safety,shared,production}]
                    [-p] [--project-dir PROJECT_DIR] [-r RULESDIR] [-R] [-s]
                    [--fix [WRITE_LIST]] [--show-relpath] [-t TAGS] [-v]
                    [-x SKIP_LIST] [--generate-ignore] [-w WARN_LIST]
                    [--enable-list ENABLE_LIST] [--nocolor] [--force-color]
                    [--exclude EXCLUDE_PATHS [EXCLUDE_PATHS ...]]
                    [-c CONFIG_FILE] [-i IGNORE_FILE]
                    [--yamllint-file YAMLLINT_FILE] [--offline | --no-offline]
                    [--version]
                    [lintables ...]

positional arguments:
  lintables             One or more files or paths. When missing it will enable auto-detection mode.

options:
  -h, --help            show this help message and exit
  -P, --list-profiles   List all profiles.
  -L, --list-rules      List all the rules.
  -T, --list-tags       List all the tags and the rules they cover.
  -f, --format {brief,full,md,json,codeclimate,quiet,pep8,sarif}
                        stdout formatting, json being an alias for codeclimate. (default: None)
  --sarif-file SARIF_FILE
                        SARIF output file
  -q                    quieter, reduce verbosity, can be specified twice.
  --profile {min,basic,moderate,safety,shared,production}
                        Specify which rules profile to be used.
  -p, --parseable       parseable output, same as '-f pep8'
  --project-dir PROJECT_DIR
                        Location of project/repository, autodetected based on location of configurat
ion file.
  -r, --rules-dir RULESDIR
                        Specify custom rule directories. Add -R to keep using embedded rules from /h
ome/docs/checkouts/readthedocs.org/user_builds/ansible-lint/checkouts/latest/src/ansiblelint/rules
  -R                    Keep default rules when using -r
  -s, --strict          Return non-zero exit code on warnings as well as errors
  --fix [WRITE_LIST]    Allow ansible-lint to perform auto-fixes, including YAML reformatting. You c
an limit the effective rule transforms (the 'write_list') by passing a keywords 'all' or 'none' or a
 comma separated list of rule ids or rule tags. YAML reformatting happens whenever '--fix' or '--fix
=' is used. '--fix' and '--fix=all' are equivalent: they allow all transforms to run. Presence of --
fix in command overrides config file value.
  --show-relpath        Display path relative to CWD
  -t, --tags TAGS       only check rules whose id/tags match these values
  -v                    Increase verbosity level (-vv for more)
  -x, --skip-list SKIP_LIST
                        only check rules whose id/tags do not match these values.             e.g: -
-skip-list=name,run-once
  --generate-ignore     Generate a text file '.ansible-lint-ignore' that ignores all found violation
s. Each line contains filename and rule id separated by a space.
  -w, --warn-list WARN_LIST
                        only warn about these rules, unless overridden in config file. Current versi
on default value is: experimental, jinja[spacing], fqcn[deep]
  --enable-list ENABLE_LIST
                        activate optional rules by their tag name
  --nocolor             disable colored output, same as NO_COLOR=1
  --force-color         Force colored output, same as FORCE_COLOR=1
  --exclude EXCLUDE_PATHS [EXCLUDE_PATHS ...]
                        path to directories or files to skip. This option is repeatable.
  -c, --config-file CONFIG_FILE
                        Specify configuration file to use. By default it will look for '.ansible-lin
t', '.ansible-lint.yml', '.ansible-lint.yaml', '.config/ansible-lint.yml', or '.config/ansible-lint.
yaml'
  -i, --ignore-file IGNORE_FILE
                        Specify ignore file to use. By default it will look for '.ansible-lint-ignor
e' or '.config/ansible-lint-ignore.txt'
  --yamllint-file YAMLLINT_FILE
                        Specify yamllint config file to use. By default it will look for '.yamllint'
, '.yamllint.yaml', '.yamllint.yml', '~/.config/yamllint/config' or environment variables XDG_CONFIG
_HOME and YAMLLINT_CONFIG_FILE.
  --offline, --no-offline
                        Disable installation of requirements.yml and schema refreshing
  --version

The following environment variables are also recognized but there is no guarantee that they will wor
k in future versions:

ANSIBLE_LINT_CUSTOM_RULESDIR: Used for adding another folder into the lookup path for new rules.

ANSIBLE_LINT_IGNORE_FILE: Define it to override the name of the default ignore file `.ansible-lint-i
gnore`

ANSIBLE_LINT_WRITE_TMP: Tells linter to dump fixes into different temp files instead of overriding o
riginal. Used internally for testing.

ANSIBLE_LINT_SKIP_SCHEMA_UPDATE: Tells ansible-lint to skip schema refresh.

ANSIBLE_LINT_NODEPS: Avoids installing content dependencies and avoids performing checks that would 
fail when modules are not installed. Far less violations will be reported.
`

### Command output[¶][3]

Ansible-lint prints output on both `stdout` and `stderr`.

* `stdout` displays rule violations.
* `stderr` displays logging and free-form messages like statistics.

Most `ansible-lint` examples use pep8 as the output format (`-p`) which is machine parseable.

Ansible-lint also print errors using their [annotation][4] format when it detects the
`GITHUB_ACTIONS=true` and `GITHUB_WORKFLOW=...` variables.

## Caching[¶][5]

For optimal performance, Ansible-lint creates caches with installed or mocked roles, collections,
and modules in the `{project_dir}/.cache` folder. The location of `{project_dir}` is passed with a
command line argument, determined by the location of the configuration file, git project top-level
directory, or user home directory.

To perform faster re-runs, Ansible-lint does not automatically clean the cache. If required you can
do this manually by simply deleting the `.cache` folder. Ansible-lint creates a new cache on the
next invocation.

You should add the `.cache` folder to the `.gitignore` file in your git repositories.

## Gradual adoption[¶][6]

For an easier gradual adoption, adopters should consider [ignore file][7] feature. This allows the
quick introduction of a linter pipeline for preventing the addition of new violations, while known
violations are ignored. Some people can work on addressing these historical violations while others
may continue to work on other maintenance tasks.

The deprecated `--progressive` mode was removed in v6.16.0 as it added code complexity and
performance overhead. It also presented several corner cases where it failed to work as expected and
caused false negatives.

## Linting playbooks and roles[¶][8]

Ansible-lint recommends following the [collection structure layout][9] whether you plan to build a
collection or not.

Following that layout assures the best integration with all ecosystem tools because it helps those
tools better distinguish between random YAML files and files managed by Ansible. When you call
`ansible-lint` without arguments, it uses internal heuristics to determine file types.

Pass the **roles** and **playbooks** that you want to lint as arguments to the `ansible-lint`
command. For example, to lint `examples/playbooks/play.yml` and `examples/roles/bobbins`, use the
following command:

`$ ansible-lint examples/playbooks/play.yml examples/roles/bobbins
WARNING  Skipped installing collection dependencies due to running in offline mode.
WARNING  Listing 6 violation(s) that are fatal
Read documentation for instructions on how to ignore specific rule violations.

# Rule Violation Summary

  1 command-instead-of-module profile:basic tags:command-shell,idiom
  1 name profile:basic tags:idiom
  1 latest profile:basic tags:idempotency
  1 no-changed-when profile:basic tags:command-shell,idempotency
  1 fqcn profile:basic tags:formatting
  1 args profile:basic tags:syntax,experimental

Failed: 5 failure(s), 1 warning(s) in 4 files processed of 4 encountered. Last profile that met the 
validation criteria was 'min'.
name[play]: All plays should be named.
examples/playbooks/play.yml:2:3

command-instead-of-module: service used in place of service module
examples/playbooks/play.yml:5 Task/Handler: A bad play

no-changed-when: Commands should not change things if nothing needs doing.
examples/playbooks/play.yml:5 Task/Handler: A bad play

args[module]: missing required arguments: repo (warning)
examples/roles/bobbins/tasks/main.yml:2 Task/Handler: Test tasks

latest[git]: Result of the command may vary on subsequent runs.
examples/roles/bobbins/tasks/main.yml:2 Task/Handler: Test tasks

fqcn[action-core]: Use FQCN for builtin module actions (git).
examples/roles/bobbins/tasks/main.yml:3:11 Use `ansible.builtin.git` or `ansible.legacy.git` instead
.
`

## Running example playbooks[¶][10]

Ansible-lint includes an `ansible-lint/examples` folder that contains example playbooks with
different rule violations and undesirable characteristics. You can run `ansible-lint` on the example
playbooks to observe Ansible-lint in action, as follows:

`$ ansible-lint --offline -p examples/playbooks/example.yml
WARNING  Skipped installing collection dependencies due to running in offline mode.
WARNING  Ignored exception from NameRule.matchyaml while processing examples/playbooks/example.yml (
playbook): 'NoneType' object has no attribute 'get'
WARNING  Listing 22 violation(s) that are fatal
Read documentation for instructions on how to ignore specific rule violations.

# Rule Violation Summary

  2 command-instead-of-module profile:basic tags:command-shell,idiom
  2 jinja profile:basic tags:formatting
  4 no-free-form profile:basic tags:syntax,risk
  1 schema profile:basic tags:core
  2 name profile:basic tags:idiom
  3 latest profile:basic tags:idempotency
  2 package-latest profile:basic tags:idempotency
  3 no-changed-when profile:basic tags:command-shell,idempotency
  3 args profile:basic tags:syntax,experimental

Failed: 19 failure(s), 3 warning(s) in 1 files processed of 1 encountered. Last profile that met the
 validation criteria was 'min'.
examples/playbooks/example.yml:1: schema[playbook][/]: $[0].tasks[13] None is not of type 'object'[/
]
examples/playbooks/example.yml:9: no-changed-when: Commands should not change things if nothing need
s doing.
examples/playbooks/example.yml:10:15: jinja[spacing][/]: Jinja2 spacing could be improved: echo {{th
is_variable}} is not set in this playbook -> echo {{ this_variable }} is not set in this playbook
examples/playbooks/example.yml:12: no-changed-when: Commands should not change things if nothing nee
ds doing.
examples/playbooks/example.yml:15: args[module][/]: missing required arguments: repo (warning)
examples/playbooks/example.yml:15: latest[git][/]: Result of the command may vary on subsequent runs
.
examples/playbooks/example.yml:18: args[module][/]: missing required arguments: repo (warning)
examples/playbooks/example.yml:18: latest[git][/]: Result of the command may vary on subsequent runs
.
examples/playbooks/example.yml:21: args[module][/]: Unsupported parameters for (basic.py) module: bo
bbins. Supported parameters include: accept_hostkey, accept_newhostkey, archive, archive_prefix, bar
e, clone, depth, dest, executable, force, gpg_allowlist, key_file, recursive, reference, refspec, re
mote, repo, separate_git_dir, single_branch, ssh_opts, track_submodules, umask, update, verify_commi
t, version (gpg_whitelist, name). (warning)
examples/playbooks/example.yml:21: no-free-form: Avoid using free-form when calling module actions. 
(ansible.builtin.git)
examples/playbooks/example.yml:24: command-instead-of-module: git used in place of git module
examples/playbooks/example.yml:24: no-changed-when: Commands should not change things if nothing nee
ds doing.
examples/playbooks/example.yml:27: command-instead-of-module: git used in place of git module
examples/playbooks/example.yml:30: latest[git][/]: Result of the command may vary on subsequent runs
.
examples/playbooks/example.yml:34:15: jinja[spacing][/]: Jinja2 spacing could be improved: {{item}} 
-> {{ item }}
examples/playbooks/example.yml:39: no-free-form: Avoid using free-form when calling module actions. 
(ansible.builtin.dnf)
examples/playbooks/example.yml:39: package-latest: Package installs should not use latest.
examples/playbooks/example.yml:42: name[missing][/]: All tasks should be named.
examples/playbooks/example.yml:42: no-free-form: Avoid using free-form when calling module actions. 
(ansible.builtin.debug)
examples/playbooks/example.yml:44: no-free-form: Avoid using free-form when calling module actions. 
(ansible.builtin.apt)
examples/playbooks/example.yml:44: package-latest: Package installs should not use latest.
examples/playbooks/example.yml:47: name[missing][/]: All tasks should be named.
`

Ansible-lint also handles playbooks that include other playbooks, tasks, handlers, or roles, as the
`examples/playbooks/include.yml` example demonstrates.

`$ ansible-lint --offline -q -p examples/playbooks/include.yml
examples/playbooks/include.yml:13:7: syntax-check[specific][/]
`

## Output formats[¶][11]

### pep8[¶][12]

`$ ansible-lint --offline -q -f pep8 examples/playbooks/norole.yml
examples/playbooks/norole.yml:5:7: syntax-check[specific][/]
`

### SARIF JSON[¶][13]

Using `--format sarif` or `--format json` the linter will output on stdout a report in [SARIF][14]

We also have an option `--sarif-file FILE` option that can make the linter dump the output to a file
while not altering its normal stdout output. This can be used in CI/CD pipelines.

SourceResult
`ansible-lint --offline -q -f sarif examples/playbooks/norole.yml
`
`{"$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0-rtm.5.json", "version":
 "2.1.0", "runs": [{"tool": {"driver": {"name": "ansible-lint", "version": "25.12.2.dev7", "informat
ionUri": "https://github.com/ansible/ansible-lint", "rules": [{"id": "syntax-check[specific]", "name
": "syntax-check[specific]", "shortDescription": {"text": "the role 'node' was not found in /home/do
cs/checkouts/readthedocs.org/user_builds/ansible-lint/checkouts/latest/examples/playbooks/roles:/hom
e/docs/checkouts/readthedocs.org/user_builds/ansible-lint/checkouts/latest/.ansible/roles:/home/docs
/.ansible/roles:/usr/share/ansible/roles:/etc/ansible/roles:/home/docs/checkouts/readthedocs.org/use
r_builds/ansible-lint/checkouts/latest/examples/playbooks"}, "defaultConfiguration": {"level": "erro
r"}, "help": {"text": ""}, "helpUri": "https://docs.ansible.com/projects/lint/rules/syntax-check/", 
"properties": {"tags": ["core", "unskippable"]}}]}}, "columnKind": "utf16CodeUnits", "results": [{"r
uleId": "syntax-check[specific]", "level": "error", "message": {"text": "the role 'node' was not fou
nd in /home/docs/checkouts/readthedocs.org/user_builds/ansible-lint/checkouts/latest/examples/playbo
oks/roles:/home/docs/checkouts/readthedocs.org/user_builds/ansible-lint/checkouts/latest/.ansible/ro
les:/home/docs/.ansible/roles:/usr/share/ansible/roles:/etc/ansible/roles:/home/docs/checkouts/readt
hedocs.org/user_builds/ansible-lint/checkouts/latest/examples/playbooks"}, "locations": [{"physicalL
ocation": {"artifactLocation": {"uri": "examples/playbooks/norole.yml", "uriBaseId": "SRCROOT"}, "re
gion": {"startLine": 5, "startColumn": 7}}}]}], "originalUriBaseIds": {"SRCROOT": {"uri": "file:///h
ome/docs/checkouts/readthedocs.org/user_builds/ansible-lint/checkouts/latest/"}}}]}
`

### Code Climate JSON[¶][15]

You can generate `JSON` reports based on the [Code Climate][16] specification as the
`examples/playbooks/norole.yml` example demonstrates.

SourceResult
`ansible-lint --offline -q -f codeclimate examples/playbooks/norole.yml
`
`[{"type": "issue", "check_name": "syntax-check[specific]", "categories": ["core", "unskippable"], "
url": "https://docs.ansible.com/projects/lint/rules/syntax-check/", "severity": "major", "descriptio
n": "the role 'node' was not found in /home/docs/checkouts/readthedocs.org/user_builds/ansible-lint/
checkouts/latest/examples/playbooks/roles:/home/docs/checkouts/readthedocs.org/user_builds/ansible-l
int/checkouts/latest/.ansible/roles:/home/docs/.ansible/roles:/usr/share/ansible/roles:/etc/ansible/
roles:/home/docs/checkouts/readthedocs.org/user_builds/ansible-lint/checkouts/latest/examples/playbo
oks", "fingerprint": "cd6fdb7a3be48e92851f6a1ec5355aef043390203ca47883d524fda5b95b36e6", "location":
 {"path": "examples/playbooks/norole.yml", "positions": {"begin": {"line": 5, "column": 7}}}}]
`

Historically `-f json` was used to generate Code Climate JSON reports but in newer versions we
switched its meaning point SARIF JSON format instead.

Warning

When possible we recommend using the [SARIF][17] format instead of the Code Climate as that one is
more complete and has a full specification and also a JSON validation schema. Code Climate format
does not expose our severity levels because we use that field to map warnings as `minor` and errors
as `major` issues.

## Specifying rules at runtime[¶][18]

By default, `ansible-lint` applies rules found in `ansible-lint/src/ansiblelint/rules`. Use the `-r
/path/to/custom-rules` option to specify the directory path to a set of custom rules. For multiple
custom rule sets, pass each set with a separate `-r` option.

You can also combine the default rules with custom rules with the `-R` option along with one or more
`-r` options.

### Including rules with tags[¶][19]

Each rule has an associated set of one or more tags. Use the `-T` option to view the list of tags
for each available rule.

You can then use the `-t` option to specify a tag and include the associated rules in the lint run.
For example, the following `ansible-lint` command applies only the rules associated with the
*idempotency* tag:

`$ ansible-lint -t idempotency playbook.yml
WARNING  Skipped installing collection dependencies due to running in offline mode.

Passed: 0 failure(s), 0 warning(s) in 1 files processed of 1 encountered. Last profile that met the 
validation criteria was 'production'.
`

The following shows the available tags in an example set of rules and the rules associated with each
tag:

`ansible-lint -T 2>/dev/null
# List of tags and rules they cover
command-shell:  # Specific to use of command and shell modules
  - risky-shell-pipe
core:  # Related to internal implementation of the linter
  - schema[ansible-lint-config]
  - schema[ansible-navigator-config]
  - schema[changelog]
  - schema[execution-environment]
  - schema[galaxy]
  - schema[inventory]
  - schema[meta-runtime]
  - schema[meta]
  - schema[molecule]
  - schema[play-argspec]
  - schema[playbook]
  - schema[requirements]
  - schema[role-arg-spec]
  - schema[rulebook]
  - schema[tasks]
  - schema[vars]
deprecations:  # Indicate use of features that are removed from Ansible
  - role-name[path]
experimental:  # Newly introduced rules, by default triggering only warnings
  - only-builtins
formatting:  # Related to code-style
  - risky-octal
idempotency:  # Possible indication that consequent runs would produce different results
  - package-latest
idiom:  # Anti-pattern detected, likely to cause undesired behavior
  - var-naming[no-jinja]
  - var-naming[no-reserved]
  - var-naming[pattern]
metadata:  # Invalid metadata, likely related to galaxy, collections or roles
  - role-name[path]
opt-in:  # Rules that are not used unless manually added to `enable_list`
  - only-builtins
risk:
  - no-free-form[raw-non-string]
  - no-free-form[raw]
security:  # Rules related to potential security issues, like exposing credentials
  - no-log-password
syntax:  # Related to wrong or deprecated syntax
  - no-free-form[raw-non-string]
  - no-free-form[raw]
unpredictability:  # Warn about code that might not work in a predictable way
  - risky-file-permissions
unskippable:  # Indicate a fatal error that cannot be ignored or disabled
  - syntax-check
yaml:  # External linter which will also produce its own rule codes
  - yaml[anchors]
  - yaml[braces]
  - yaml[brackets]
  - yaml[colons]
  - yaml[commas]
  - yaml[comments-indentation]
  - yaml[comments]
  - yaml[document-end]
  - yaml[document-start]
  - yaml[empty-lines]
  - yaml[empty-values]
  - yaml[float-values]
  - yaml[hyphens]
  - yaml[indentation]
  - yaml[key-duplicates]
  - yaml[key-ordering]
  - yaml[line-length]
  - yaml[new-line-at-end-of-file]
  - yaml[new-lines]
  - yaml[octal-values]
  - yaml[quoted-strings]
  - yaml[trailing-spaces]
  - yaml[truthy]
`

### Excluding rules with tags[¶][20]

To exclude rules by identifiers or tags, use the `-x SKIP_LIST` option. For example, the following
command applies all rules except those with the *formatting* and *metadata* tags:

`$ ansible-lint -x formatting,metadata playbook.yml
`

### Ignoring rules[¶][21]

To only warn about rules, use the `-w WARN_LIST` option. For example, the following command displays
only warns about violations with rules associated with the `experimental` tag:

`$ ansible-lint -w experimental playbook.yml
`

By default, the `WARN_LIST` includes the `['experimental']` tag. If you define a custom `WARN_LIST`
you must add `'experimental'` so that Ansible-lint does not fail against experimental rules.

## Muting warnings to avoid false positives[¶][22]

Not all linting rules are precise, some are general rules of thumb. Advanced *git*, *yum* or *apt*
usage, for example, can be difficult to achieve in a playbook. In cases like this, Ansible-lint can
incorrectly trigger rule violations.

To disable rule violations for specific tasks, and mute false positives, add `# noqa: [rule_id]` to
the end of the line. It is best practice to add a comment that explains why rules are disabled.

You can add the `# noqa: [rule_id]` comment to the end of any line in a task. You can also skip
multiple rules with a space-separated list.

`- name: This task would typically fire git-latest and partial-become rules
  become_user: alice # noqa: git-latest partial-become
  ansible.builtin.git: src=/path/to/git/repo dest=checkout
`

If the rule is line-based, `# noqa: [rule_id]` must be at the end of the line.

`- name: This would typically fire jinja[spacing]
  get_url:
    url: http://example.com/file.conf
    dest: "{{dest_proj_path}}/foo.conf" # noqa: jinja[spacing]
`

If you want Ansible-lint to skip a rule entirely, use the `-x` command line argument or add it to
`skip_list` in your configuration.

The least preferred method of skipping rules is to skip all task-based rules for a task, which does
not skip line-based rules. You can use the `skip_ansible_lint` tag with all tasks, for example:

`- name: This would typically fire no-free-form
  command: warn=no chmod 644 X

- name: This would typically fire git-latest
  git: src=/path/to/git/repo dest=checkout
  tags:
    - skip_ansible_lint
`

## Applying profiles[¶][23]

Ansible-lint profiles allow content creators to progressively improve the quality of Ansible
playbooks, roles, and collections.

During early development cycles, you need Ansible-lint rules to be less strict. Starting with the
minimal profile ensures that Ansible can load your content. As you move to the next stage of
developing content, you can gradually apply profiles to avoid common pitfalls and brittle
complexity. Then, when you are ready to publish or share your content, you can use the `shared` and
`production` profiles with much stricter rules. These profiles harden security, guarantee
reliability, and ensure your Ansible content is easy for others to contribute to and use.

Note

Tags such as `opt-in` and `experimental` do not take effect for rules that are included in profiles,
directly or indirectly. If a rule is in a profile, Ansible-lint applies that rule to the content.

After you install and configure `ansible-lint`, you can apply profiles as follows:

1. View available profiles with the `--list-profiles` flag.
`ansible-lint --list-profiles
`

1. Specify a profile with the `--profile` parameter to lint your content with those rules, for
   example:
2. Enforce standard styles and formatting with the `basic` profile.
`ansible-lint --profile=basic
`

* Ensure automation consistency, reliability, and security with the `safety` profile.
`ansible-lint --profile=safety
`

## Vaults[¶][24]

As ansible-lint executes ansible, it also needs access to encrypted secrets. If you do not give
access to them or you are concerned about security implications, you should consider refactoring
your code to allow it to be linted without access to real secrets:

* Configure dummy fallback values that are used during linting, so Ansible will not complain about
  undefined variables.
* Exclude the problematic files from the linting process.
`---
# Example of avoiding undefined variable error
foo: "{{ undefined_variable_name | default('dummy') }}"
`

Keep in mind that a well-written playbook or role should allow Ansible's syntax check from passing
on it, even if you do not have access to the vault.

Internally ansible-lint runs `ansible-playbook --syntax-check` on each playbook and also on roles.
As ansible-code does not support running syntax-check directly on roles, the linter will create
temporary playbooks that only include each role from your project. You will need to change the code
of the role in a way that it does not produce syntax errors when called without any variables or
arguments. This usually involves making use of `defaults/` but be sure that you fully understand
[variable precedence][25].

## Dependencies and requirements[¶][26]

Ansible-lint will recognize `requirements.yml` files used for runtime and testing purposes and
install them automatically. Valid locations for these files are:

* [`requirements.yml`][27]
* `roles/requirements.yml`
* `collections/requirements.yml`
* `tests/requirements.yml`
* `tests/integration/requirements.yml`
* `tests/unit/requirements.yml`
* [`galaxy.yml`][28]

[1]: #using
[2]: #using-commands
[3]: #command-output
[4]: https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#setting
-an-error-message
[5]: #caching
[6]: #gradual-adoption
[7]: ../configuring/#ignoring-rules-for-entire-files
[8]: #linting-playbooks-and-roles
[9]: https://docs.ansible.com/projects/ansible-core/devel/dev_guide/developing_collections_structure
.html#collection-structure
[10]: #running-example-playbooks
[11]: #output-formats
[12]: #pep8
[13]: #sarif-json
[14]: https://docs.oasis-open.org/sarif/sarif/v2.1.0/csprd01/sarif-v2.1.0-csprd01.html
[15]: #code-climate-json
[16]: https://github.com/codeclimate/platform/blob/master/spec/analyzers/SPEC.md#data-types
[17]: #sarif-json
[18]: #specifying-rules-at-runtime
[19]: #including-rules-with-tags
[20]: #excluding-rules-with-tags
[21]: #ignoring-rules
[22]: #muting-warnings-to-avoid-false-positives
[23]: #applying-profiles
[24]: #vaults
[25]: https://docs.ansible.com/projects/ansible/latest/playbook_guide/playbooks_variables.html#under
standing-variable-precedence
[26]: #dependencies-and-requirements
[27]: https://docs.ansible.com/projects/ansible/latest/galaxy/user_guide.html#installing-roles-and-c
ollections-from-the-same-requirements-yml-file
[28]: https://docs.ansible.com/projects/ansible/latest/dev_guide/collections_galaxy_meta.html
