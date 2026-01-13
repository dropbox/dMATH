Ask

# Homepage

SonarQube provides automated code quality and security reviews, delivering actionable intelligence
that helps developers build better and faster.

### What is SonarQube Server?

SonarQube Server is an industry-standard on-premises automated code review and static analysis tool
designed to detect coding issues in [30+ languages][1], [frameworks, and IaC platforms][2]. By
integrating directly with your CI pipeline (see the [Overview][3] page) or on one of our supported
DevOps platforms, your code is checked against an extensive set of rules that cover many attributes
of code, such as maintainability, reliability, and security issues on each merge/pull request.

As a core element of the SonarQube solution, SonarQube Server completes the analysis loop to help
you deliver code that meets high-quality standards.

Please see the [Try out SonarQube Server][4] page to learn how to get started. For a
Software-as-a-Service (SaaS) cloud-based tool, see [SonarQube Cloud][5].

### Achieving high quality code

SonarQube sets high standards for all code that results in secure, reliable, and maintainable
software that is essential to maintaining a healthy codebase. This applies to all code: source code,
test code, infrastructure as code, glue code, scripts, and more.

All new code, whether added or recently modified, should adhere to quality standards. SonarQube for
IDE achieves this by providing automated code reviews that alert you to potential issues within your
new code. This helps you maintain high standards and focus on code quality, ultimately leading to a
healthier codebase over time.

SonarQube Server comes with a built-in quality profiles designed for each supported language, called
the **Sonar way** profile, see [Understanding quality profiles][6]. The **Sonar way** activates a
set of rules that should be applicable to most projects and is a starting point to help you
implement good practices in your organization.

### The SonarQube solution

SonarQube is designed to help you achieve a state of high quality code. By linking SonarQube for IDE
([VS Code][7], [Intellij][8], [Visual Studio][9] , [Eclipse][10]) with SonarQube Cloud or SonarQube
Server, the automated code analysis and reviews are performed at every stage of the development
process. We call this the SonarQube solution. This means your project settings, new code
definitions, and quality profiles are applied locally to an analysis in the IDE. Your project
settings, new code definitions, and the quality profiles managed in SonarQube (Server, Cloud) are
applied locally to an analysis in the IDE.

* SonarQube for IDE ([VS Code][11], [Intellij][12], [Visual Studio][13] , [Eclipse][14]) brings
  automated code reviews directly into your development environment, helping you catch issues as you
  write code. By providing immediate feedback, it enables engineers to identify and fix problems
  before they even commit, ensuring cleaner, higher-quality code from the start.
* SonarQube delivers powerful static code analysis by thoroughly reviewing each pull request before
  it’s merged. This proactive approach adds an essential layer of protection, ensuring code quality
  and preventing issues from entering your codebase. See the [introduction to PR analysis][15] on
  SonarQube Server and [Pull request analysis][16] on SonarQube Cloud.
* Finally, SonarQube Server and SonarQube Cloud seamlessly integrate into your CI/CD pipeline,
  analyzing code on every build. By leveraging [quality profiles][17] and [quality gates][18], they
  automatically block code with issues from being released to production, ensuring only
  maintainable, reliable, and secure code makes it through.

The SonarQube solution helps you incorporate a proper methodology by helping engineers pay attention
to new code. Focusing on writing high quality new code during development ensures that all code
released for production will be incrementally improved over time.

### Connected Mode

Connected Mode joins SonarQube Server with SonarQube for IDE to deliver the full SonarQube solution.
While in Connected Mode, SonarQube Server sends notifications to SonarQube for IDE when a quality
gate changes or a new issue is assigned to the user. Smart notifications can be enabled or disabled
from the SonarQube for IDE interface while creating or editing the connection settings.
Additionally, SonarQube for IDE helps engineers focus on writing high quality code by using the new
code definition from the server. Be sure to check out all of the benefits of [Connected mode][19].

### Getting started

Now that you’ve heard about how [SonarQube Server][20] can help you write high quality code, you are
ready to try out SonarQube Server for yourself. You can run a local non-production instance of
SonarQube Server and the initial project analysis. Installing a local instance gets you up and
running quickly, so you can experience SonarQube Server firsthand. Then, when you’re ready to set up
SonarQube Server in production, you’ll need to follow this [Introduction][21] to installation before
configuring your first code analysis.

The [Project analysis setup][22] section explains how to connect your scanner to your CI pipeline
and provides instructions for analyzing your project’s branches and pull requests.

Here is a page with everything you need to [Try out SonarQube Server][23].

### Learn more

Check out the entire suite of Sonar products: [SonarQube Server][24], [SonarQube Cloud][25], and
[SonarQube for IDE][26] and browse a full list of [Sonar Rules and Rule Descriptions][27] available
for static code analysis.

Then, have a look at how to fix issues detected by SonarQube for IDE in

* VS Code: [Fixing issues][28]
* IntelliJ: [Fixing issues][29]
* Visual studio: [Fixing issues][30]
* Eclipse: [Fixing issues][31]

#### More getting started resources

* [Introduction][32] to server installation and setup
* [Creating and importing projects][33]
* [Introduction][34]
* [Managing portfolios][35]

### Staying connected

If you need help, visit our [online community][36] to search for answers and reach out with
questions!

[NextTry out SonarQube Server][37]

Last updated 12 days ago

Was this helpful?

[1]: https://rules.sonarsource.com/
[2]: https://rules.sonarsource.com/
[3]: /sonarqube-server/2025.4/analyzing-source-code/ci-integration/overview
[4]: /sonarqube-server/2025.4/try-out-sonarqube
[5]: https://docs.sonarsource.com/sonarqube-cloud/
[6]: /sonarqube-server/2025.4/quality-standards-administration/managing-quality-profiles/understandi
ng-quality-profiles
[7]: https://docs.sonarsource.com/sonarqube-for-vs-code/
[8]: https://docs.sonarsource.com/sonarqube-for-intellij/
[9]: https://docs.sonarsource.com/sonarqube-for-visual-studio/
[10]: https://docs.sonarsource.com/sonarqube-for-eclipse/
[11]: https://docs.sonarsource.com/sonarqube-for-vs-code/
[12]: https://docs.sonarsource.com/sonarqube-for-intellij/
[13]: https://docs.sonarsource.com/sonarqube-for-visual-studio/
[14]: https://docs.sonarsource.com/sonarqube-for-eclipse/
[15]: /sonarqube-server/2025.4/analyzing-source-code/pull-request-analysis
[16]: /sonarqube-cloud/improving/pull-request-analysis
[17]: /sonarqube-server/2025.4/quality-standards-administration/managing-quality-profiles/understand
ing-quality-profiles
[18]: /sonarqube-server/2025.4/quality-standards-administration/managing-quality-gates/introduction-
to-quality-gates
[19]: /sonarqube-server/2025.4/user-guide/connected-mode
[20]: https://www.sonarsource.com/products/sonarqube/
[21]: /sonarqube-server/2025.4/server-installation/introduction
[22]: /sonarqube-server/2025.4/analyzing-source-code/overview
[23]: /sonarqube-server/2025.4/try-out-sonarqube
[24]: https://www.sonarsource.com/products/sonarqube/
[25]: https://www.sonarsource.com/products/sonarcloud/
[26]: https://www.sonarsource.com/products/sonarlint/
[27]: http://rules.sonarsource.com/
[28]: /sonarqube-for-vs-code/using/fixing-issues
[29]: /sonarqube-for-intellij/using/fixing-issues
[30]: /sonarqube-for-visual-studio/using/fixing-issues
[31]: /sonarqube-for-eclipse/using/fixing-issues
[32]: /sonarqube-server/2025.4/server-installation/introduction
[33]: /sonarqube-server/2025.4/project-administration/creating-and-importing-projects
[34]: /sonarqube-server/2025.4/quality-standards-administration/managing-quality-profiles/introducti
on
[35]: /sonarqube-server/2025.4/project-administration/managing-portfolios
[36]: https://community.sonarsource.com/c/sq/10
[37]: /sonarqube-server/2025.4/try-out-sonarqube
