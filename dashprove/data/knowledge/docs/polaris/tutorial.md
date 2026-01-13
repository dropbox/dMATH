# [#][1] Infrastructure as Code

> Want to see results for all your IaC repos in one place? Check out [Fairwinds Insights (opens new
> window)][2]

Polaris can be used on the command line to audit local Kubernetes manifests stored in YAML files.
This is particularly helpful for running Polaris against your infrastructure-as-code as part of a
CI/CD pipeline. Use the available [command line flags][3] to cause CI/CD to fail if your Polaris
score drops below a certain threshold, or if any danger-level issues arise.

## [#][4] Install the CLI

To run Polaris against your YAML manifests, e.g. as part of a Continuous Integration process, you'll
need to install the CLI.

Binary releases can be downloaded from the [releases page (opens new window)][5] or can be installed
with [Homebrew (opens new window)][6]:

`brew tap FairwindsOps/tap
brew install FairwindsOps/tap/polaris
polaris version
`

## [#][7] Checking Infrastructure as Code files

You can audit Kubernetes YAML files by running:

`polaris audit --audit-path ./deploy/ --format=pretty
`

This will print out any issues Polaris finds in your manifests.

Polaris can only check raw YAML manifests. If you'd like to check a Helm template, you can run `helm
template` to generate a manifest that Polaris can check.

## [#][8] Fixing Issues

Polaris can automatically fix many of the issues it finds. For example, you can run

`polaris fix --files-path ./deploy/ --checks=all
`

to fix any issues inside the `deploy` directory. Polaris may leave comments next to some changes
(e.g. liveness and readiness probes) prompting the user to set them to something more appropriate
given the context of their application.

Note that not all issues can be automatically fixed.

Currently only raw YAML manifests can be mutated. Helm charts etc. still need to be changed
manually.

## [#][9] Running in a CI pipeline

### [#][10] Set minimum score for an exit code

You can tell the CLI to set an exit code if it detects certain issues with your YAML files. For
example, to fail if polaris detects *any* danger-level issues, or if the score drops below 90%:

`polaris audit --audit-path ./deploy/ \
  --set-exit-code-on-danger \
  --set-exit-code-below-score 90
`

### [#][11] Pretty-print results

By default, results are output as JSON. You can get human-readable output with the `--format=pretty`
flag:

`polaris audit --audit-path ./deploy/ \
  --format=pretty
`

You can also disable colors and emoji:

`polaris audit --audit-path ./deploy/ \
  --format=pretty \
  --color=false
`

### [#][12] Output only showing failed tests

The CLI to gives you ability to display results containing only failed tests. For example:

`polaris audit --audit-path ./deploy/ \
  --only-show-failed-tests true
`

### [#][13] Audit Helm Charts

You can audit helm charts using the `--helm-chart` and `--helm-values` flags:

`polaris audit \
  --helm-chart ./deploy/chart \
  --helm-values ./deploy/chart/values.yml
`

### [#][14] As Github Action

#### [#][15] Setup polaris action

This action downloads a version of [polaris (opens new window)][16] and adds it to the path. It
makes the [polaris cli (opens new window)][17] ready to use in following steps of the same job.

[#][18] Inputs [#][19] `version`

The release version to fetch. This has to be in the form `<tag_name>`.

[#][20] Outputs [#][21] `version`

The version number of the release tag.

[#][22] Example usage
`uses: fairwindsops/polaris/.github/actions/setup-polaris@master
with:
  version: 5.0.0
`

Example inside a job:

`steps:
  - uses: actions/checkout@v2
  - name: Setup polaris
    uses: fairwindsops/polaris/.github/actions/setup-polaris@master
    with:
      version: 5.0.0

  - name: Use command
    run: polaris version
`
[Help us improve this page][23] (opens new window)

← [ Admission Controller ][24] [ CLI Options ][25] →

[Learn more about Fairwinds][26] [Try Fairwinds Insights][27]
[Privacy Policy][28]

[1]: #infrastructure-as-code
[2]: https://www.fairwinds.com/fairwinds-polaris-upgrade
[3]: #running-in-a-ci-pipeline
[4]: #install-the-cli
[5]: https://github.com/fairwindsops/polaris/releases
[6]: https://brew.sh/
[7]: #checking-infrastructure-as-code-files
[8]: #fixing-issues
[9]: #running-in-a-ci-pipeline
[10]: #set-minimum-score-for-an-exit-code
[11]: #pretty-print-results
[12]: #output-only-showing-failed-tests
[13]: #audit-helm-charts
[14]: #as-github-action
[15]: #setup-polaris-action
[16]: https://github.com/FairwindsOps/polaris
[17]: https://polaris.docs.fairwinds.com/infrastructure-as-code
[18]: #inputs
[19]: #version
[20]: #outputs
[21]: #version-2
[22]: #example-usage
[23]: https://github.com/FairwindsOps/polaris/edit/master/docs/infrastructure-as-code.md
[24]: /admission-controller/
[25]: /cli/
[26]: https://fairwinds.com
[27]: https://fairwinds.com/insights
[28]: https://www.fairwinds.com/privacy-policy
