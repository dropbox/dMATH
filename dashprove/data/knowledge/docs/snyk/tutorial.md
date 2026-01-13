[Edit][1]

1. [Discover Snyk][2]

# Getting started

Ensure you use languages, package managers, and frameworks supported by Snyk. See [Supported
languages, package managers, and frameworks][3].

## Supported browsers

Snyk does not support Microsoft Internet Explorer.

Snyk supports the latest versions of the following web browsers:

* [Chrome][4]
* [Edge][5]
* [Firefox][6]
* [Safari][7] (except for [Opening Fix PR][8])

Snyk requires Javascript to be enabled on your browser.

To start performing basic tasks in the Snyk application:

## Create or log in to a Snyk account

To create a free account or sign up for a pricing plan, navigate to [snyk.io][9]. For details, see
[Snyk Pricing Plans][10].

If your company has an existing Snyk account and uses single sign-on (SSO), use the SSO link
provided by your administrators.

If your company requires an invitation to use Snyk, when you log in for the first time, you may see
a list of Organizations, which in Snyk control access to Projects. To request access to an
Organization, select the name of an Organization Admin in order to request access.

If you log in with a different authentication provider from the one your company uses for the Snyk
account, you create a new account. You will not be logged in to the correct Organization for your
company.

When you log in to the Snyk Web UI, Snyk shows your preferred (default) Organization. Snyk also uses
the settings for your preferred Organization when you test a Project locally using the CLI.

## Set up a Snyk integration

For Snyk to know where to scan, you must provide it with access to your environment. The type of
integration you need depends on what systems you use, what you want to scan, and where you want to
add the integrations - [Organization][11] or [Group][12]. For information about available
integrators, see [Snyk SCM integrations][13] and [Integrate with Snyk][14].

To scan your code, you must first integrate Snyk with the repository holding that code.

### Guided process

After creating a Snyk account, you can follow the optional getting-started walkthrough prompts to
provide information and help Snyk guide your experience. This includes choosing an integration
method, setting access permissions, configuring automation settings, and authenticating that
integration.

Alternatively, if you want to scan your code without authenticating to your source code repository,
you can select the CLI integration. This allows you to run scans from your local machine and upload
results to your Organization in Snyk.

### Manual process

You can add an integration to Snyk manually at any point from the Snyk Web UI. To do this, navigate
to **Integrations** > **Source Control**. For more information, see [Integrate with Snyk][15].

If an integration is already configured for your Organization, it is marked as **Configured**.

## Obtain and use your Snyk API token

Before authenticating, be sure you have set your region properly. For details, see [Regional hosting
and data residency][16], which has the [list of regional URLs][17].

Your Snyk API token is a personal token available under your user profile. The Snyk API token is
associated with your Snyk Account and not with a specific Organization.

Free and Team plan and trial users have access only to this personal token under the user profile.
The personal token can be used to authenticate with the Snyk CLI running on a local or a build
machine and an IDE when you are setting a token manually. Use a personal token with caution if you
are authenticating for CI/CD or with the API, which is available for Enterprise plan users only.

To obtain your personal Snyk API token:

1. Log in to Snyk and navigate to your personal account settings.
2. In your **General** settings, under API Token, select **click to show**.
3. Highlight and copy your API key.

If you want a new API token, select **Revoke & Regenerate**, but be aware that this will make the
previous API token invalid.

For information on when to use an API token and when to use a service account token, available to
Enterprise plan users only, see [Authentication for API][18].

## Import a Project to scan and identify issues

Snyk Projects are items that Snyk scans for issues, for example, a manifest file listing your
open-source dependencies.

When you import a Project, Snyk scans that imported Project, and displays the results for you to
review.

Importing a Project also does the following:

* Sets Snyk to run a regular scan on that Project for issues. See [Usage settings][19].
* Initiates some automation, especially default Snyk tests on pull and merge requests, which help
  prevent vulnerabilities from being added to the Project. This automation fails builds according to
  your conditions and can be disabled or customized in your [integration settings][20].

## Set up Snyk Essentials

Snyk Essentials is available only with Enterprise plans. For more information, see [plans and
pricing][21].

Snyk Essentials enables Application Security teams to implement, manage, and scale a modern,
high-performing, developer security program. It covers use cases under Application Security Posture
Management (ASPM).

For more information, see [Snyk Essentials][22].

## Review results and fix your issues

After you have imported a Project, and Snyk has scanned that Project for issues, you can view the
results of your scan and take action to fix issues. You can see the number of issues found, grouped
by severity level (**C**ritical, **H**igh, **M**edium or **L**ow). For details, see [Severity
levels][23].

The scan results and available actions depend on the type of Project you scan:

* Open-source libraries: see [Snyk Open Source][24].
* Application code: see [Snyk Code][25].
* Container images: see [Snyk Container][26].
* Infrastructure as Code (IaC), Kubernetes, Helm and Terraform configuration files and cloud
  misconfigurations: see [Snyk IaC][27].
[PreviousWhat's new?][28][NextSnyk release process][29]

Last updated 4 months ago

Was this helpful?

[1]: https://github.com/snyk/user-docs/blob/main/docs/discover-snyk/getting-started/README.md
[2]: /discover-snyk
[3]: /supported-languages/supported-languages-package-managers-and-frameworks
[4]: https://www.google.com/chrome/
[5]: https://www.microsoft.com/en-us/edge?form=MA13FJ
[6]: https://www.mozilla.org/en-US/firefox/new/
[7]: https://www.apple.com/safari/
[8]: /scan-with-snyk/pull-requests/snyk-pull-or-merge-requests
[9]: https://snyk.io/
[10]: /implementation-and-setup/enterprise-implementation-guide/trial-limitations
[11]: /integrations/integrate-with-snyk
[12]: /integrations/integrate-with-snyk
[13]: /developer-tools/scm-integrations/organization-level-integrations
[14]: /integrations/integrate-with-snyk
[15]: /integrations/integrate-with-snyk
[16]: /snyk-data-and-governance/regional-hosting-and-data-residency
[17]: /snyk-data-and-governance/regional-hosting-and-data-residency#regional-urls
[18]: /snyk-api/authentication-for-api
[19]: /snyk-platform-administration/groups-and-organizations/usage-settings
[20]: /developer-tools/scm-integrations/organization-level-integrations
[21]: https://snyk.io/plans/
[22]: /scan-with-snyk/snyk-essentials
[23]: /manage-risk/prioritize-issues-for-fixing/severity-levels
[24]: /scan-with-snyk/snyk-open-source
[25]: /scan-with-snyk/snyk-code
[26]: /scan-with-snyk/snyk-container/scan-container-images
[27]: /scan-with-snyk/snyk-iac
[28]: /discover-snyk/whats-new
[29]: /discover-snyk/getting-started/snyk-release-process
