* [Support Center][1]
* [Documentation][2]
* [Desktop editions][3]
* [Running scans][4]

Professional

# Running scans as part of your manual testing workflow

* **Last updated: ** December 16, 2025
* **Read time: ** 2 Minutes

Burp Scanner is a web vulnerability scanning tool built into Burp Suite Professional. You can use
Burp Scanner to automatically map the attack surface and identify vulnerabilities in both web
applications and APIs. This streamlines your workflow by automating repetitive tasks, freeing you to
use your time and expertise on more complex manual tasks.

You can run different types of web application scans to support a wide range of use cases:

* **Crawl** - Automatically map the application's attack surface, saving you from having to manually
  navigate through the whole application, clicking every link and submitting every form.
* **Full crawl and audit** - Automatically map the attack surface and probe discovered requests for
  vulnerabilities. Burp Scanner handles the more repetitive auditing tasks so you can concentrate on
  more sophisticated manual testing.
* **Audit selected items** - Automatically probe for issues in one or more requests that you think
  may be vulnerable. This enables you to test for a wide range of vulnerabilities in seconds, rather
  than hours.

Burp's AI-powered Explore Issue feature enables you to automate follow-up testing on vulnerabilities
that Burp Scanner identifies. This can help you uncover additional attack vectors and generate
proof-of-concept exploits automatically.

If Burp Scanner discovers any API definitions in a web application scan, it parses the definition,
then audits the discovered endpoints. For more information about which API formats Burp Scanner
supports, see [Requirements for API scanning][5].

Burp Scanner also offers an API-only scanning option for when you need to do a standalone scan based
on an OpenAPI definition, SOAP WSDL, or Postman Collection.

#### Related pages

This section explains how to run and configure scans in Burp Suite Professional. For information on
how to create and manage scans in Burp Suite DAST, see [Working with scans][6].

For information on how Burp Scanner works under the hood, see the [Burp Scanner][7] documentation.

#### In this section

* [Scanning web applications][8]
* [Running API-only scans][9]
* [Live tasks][10]
* [Setting the scan scope][11]
* [Configuring scans][12]
* [Adding custom scan checks][13]
* [Adding extension scan checks][14]
* [Viewing scan results][15]
* [Exploring issues with AI][16]
* [Reporting scan results][17]
* [Configuring application logins][18]
* [Managing resource pools][19]

[1]: https://portswigger.net/support
[2]: /burp/documentation
[3]: /burp/documentation/desktop
[4]: /burp/documentation/desktop/running-scans
[5]: /burp/documentation/scanner/api-scanning-reqs
[6]: /burp/documentation/dast/user-guide/working-with-scans
[7]: /burp/documentation/scanner
[8]: /burp/documentation/desktop/running-scans/webapp-scans
[9]: /burp/documentation/desktop/running-scans/api-scans
[10]: /burp/documentation/desktop/running-scans/live-tasks
[11]: /burp/documentation/desktop/running-scans/setting-pro-scope
[12]: /burp/documentation/desktop/running-scans/configuring-scans
[13]: /burp/documentation/desktop/running-scans/custom-checks
[14]: /burp/documentation/desktop/running-scans/extension-checks
[15]: /burp/documentation/desktop/running-scans/results
[16]: /burp/documentation/desktop/running-scans/explore-issue-with-ai
[17]: /burp/documentation/desktop/running-scans/reporting
[18]: /burp/documentation/desktop/running-scans/configuring-app-logins
[19]: /burp/documentation/desktop/running-scans/managing-resource-pools
