[Polaris Logo]

### Polaris is an open source policy engine for Kubernetes

Polaris is an open source policy engine for Kubernetes that validates and remediates resource
configuration. It includes 30+ built in configuration policies, as well as the ability to build
custom policies with JSON Schema. When run on the command line or as a mutating webhook, Polaris can
automatically remediate issues based on policy criteria.

Polaris can be run in three different modes:

* As a [dashboard][1] - Validate Kubernetes resources against policy-as-code.
* As an [admission controller][2] - Automatically reject or modify workloads that don't adhere to
  your organization's policies.
* As a [command-line tool][3] - Incorporate policy-as-code into the CI/CD process to test local YAML
  files.
  
  [Polaris Architecture]

## [#][4] Join the Fairwinds Open Source Community

The goal of the Fairwinds Community is to exchange ideas, influence the open source roadmap, and
network with fellow Kubernetes users. [Chat with us on Slack (opens new window)][5]

## [#][6] Other Projects from Fairwinds

Enjoying Polaris? Check out some of our other projects:

* [Goldilocks (opens new window)][7] - Right-size your Kubernetes Deployments by compare your memory
  and CPU settings against actual usage
* [Pluto (opens new window)][8] - Detect Kubernetes resources that have been deprecated or removed
  in future versions
* [Nova (opens new window)][9] - Check to see if any of your Helm charts have updates available
* [rbac-manager (opens new window)][10] - Simplify the management of RBAC in your Kubernetes
  clusters

Or [check out the full list (opens new window)][11]

## [#][12] Fairwinds Insights

If you're interested in running Polaris in multiple clusters, tracking the results over time,
integrating with Slack, Datadog, and Jira, or unlocking other functionality, check out [Fairwinds
Insights (opens new window)][13], a platform for auditing and enforcing policy in Kubernetes
clusters.

[Help us improve this page][14] (opens new window)
[Learn more about Fairwinds][15] [Try Fairwinds Insights][16]
[Privacy Policy][17]

[1]: /dashboard
[2]: /admission-controller
[3]: /infrastructure-as-code
[4]: #join-the-fairwinds-open-source-community
[5]: https://join.slack.com/t/fairwindscommunity/shared_invite/zt-2na8gtwb4-DGQ4qgmQbczQyB2NlFlYQQ
[6]: #other-projects-from-fairwinds
[7]: https://github.com/FairwindsOps/Goldilocks
[8]: https://github.com/FairwindsOps/Pluto
[9]: https://github.com/FairwindsOps/Nova
[10]: https://github.com/FairwindsOps/rbac-manager
[11]: https://www.fairwinds.com/open-source-software?utm_source=polaris&utm_medium=polaris&utm_campa
ign=polaris
[12]: #fairwinds-insights
[13]: https://fairwinds.com/insights
[14]: https://github.com/FairwindsOps/polaris/edit/master/docs/README.md
[15]: https://fairwinds.com
[16]: https://fairwinds.com/insights
[17]: https://www.fairwinds.com/privacy-policy
