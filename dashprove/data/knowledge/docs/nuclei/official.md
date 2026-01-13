## [​][1]
## What is **Nuclei?**

Nuclei is a fast vulnerability scanner designed to probe modern applications, infrastructure, cloud
platforms, and networks, aiding in the identification and mitigation of exploitable vulnerabilities.
At its core, Nuclei uses templates—expressed as straightforward YAML files, that delineate methods
for detecting, ranking, and addressing specific security flaws. Each template delineates a possible
attack route, detailing the vulnerability, its severity, priority rating, and occasionally
associated exploits. This template-centric methodology ensures Nuclei not only identifies potential
threats, but pinpoints exploitable vulnerabilities with tangible real-world implications. New to
scanners and Nuclei? Try it out today with a quick example through our [Getting Started][2].

## [​][3]
## What are Nuclei’s features?

───────────────────┬────────────────────────────────────────────────────────────────────────────────
Feature            │Description                                                                     
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Extensive Template│Nuclei offers a vast collection of community-powered templates for targeted     
Library][4]        │scans of various vulnerabilities and attack vectors.                            
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Versatile Target  │Support for various target specification options, such as URLs, IP ranges, ASN  
Specification][5]  │range, and file input, allowing flexibility in defining the scanning scope.     
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Bulk Scanning][6] │Perform bulk scanning by specifying multiple targets at once, enabling efficient
                   │scanning of a large number of assets or websites.                               
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Flexible          │Customize scanning templates to fit specific needs, allowing tailored scanning  
Customization][7]  │and focusing on relevant security checks.                                       
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Parallel          │Supports parallel scanning, reducing scanning time and improving efficiency,    
Scanning][8]       │especially for large-scale targets.                                             
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Comprehensive     │Generates detailed reports with actionable insights, including vulnerability    
Reporting          │details, severity levels, affected endpoints, and suggested remediation steps.  
`cloud`][9]        │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Integration with  │Seamlessly integrate Nuclei into CI/CD pipelines for automated security testing 
CI/CD              │as part of the development and deployment process.                              
Pipelines][10]     │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[CI/CD Integration │Actively maintained and developed by the ProjectDiscovery team, introducing new 
`cloud`][11]       │features, bug fixes, and enhancements to provide an up-to-date scanning         
                   │framework.                                                                      
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Ticketing         │Two-way ticketing integration with Jira, Splunk, and many others to easily      
integration        │remediate and retest vulnerabilities.                                           
`cloud`][12]       │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Customizable      │Configure the output format of Nuclei’s scan results to suit your needs,        
Output Format][13] │including options for JSON, YAML, and more.                                     
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Dynamic           │Utilize dynamic variables in templates to perform parameterized scanning,       
Variables][14]     │enabling versatile and flexible scanning configurations.                        
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Inclusion and     │Apply inclusion and exclusion filters to specify targets, reducing scanning     
Exclusion          │scope and focusing on specific areas of interest.                               
Filters][15]       │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Authentication    │Nuclei supports various authentication mechanisms, including HTTP basic         
Support][16]       │authentication, JWT token authentication, and more.                             
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[Embedding custom  │Execute custom code within Nuclei templates to incorporate user-defined logic,  
code in            │perform advanced scanning actions, and more.                                    
templates][17]     │                                                                                
───────────────────┼────────────────────────────────────────────────────────────────────────────────
[AI-Powered        │Generate and run vulnerability templates on-the-fly using natural language      
Template           │descriptions powered by ProjectDiscovery’s AI capabilities.                     
Generation][18]    │                                                                                
───────────────────┴────────────────────────────────────────────────────────────────────────────────

## [​][19]
## How can I use Nuclei?

The global security community, including numerous researchers and engineers, actively contributes to
the Nuclei template ecosystem. With over 6500 templates contributed thus far, Nuclei is continuously
updated with real-world exploits and cutting-edge attack vectors. Nuclei templates support scanning
for critical issues such as the Log4j vulnerability and RCEs that impact vendors such as GitLab,
Cisco, F5, and many others. Nuclei has dozens of use cases, including:

───────────────────────────────┬────────────────────────────────────────────────────────────────────
Use Case                       │Description                                                         
───────────────────────────────┼────────────────────────────────────────────────────────────────────
Web Application Security       │Identifies common web vulnerabilities with community-powered        
                               │templates.                                                          
───────────────────────────────┼────────────────────────────────────────────────────────────────────
Infrastructure Security        │Audits server configurations, open ports, and insecure services for 
                               │security issues.                                                    
───────────────────────────────┼────────────────────────────────────────────────────────────────────
API Security Testing `alpha`   │Tests APIs against known vulnerabilities and misconfigurations.     
───────────────────────────────┼────────────────────────────────────────────────────────────────────
(CI/CD) Security               │Integrates into CI/CD pipelines to minimize vulnerability resurface 
                               │into production.                                                    
───────────────────────────────┼────────────────────────────────────────────────────────────────────
Third-party Vendor Assessment  │Evaluates the security of third-party vendors by scanning their     
                               │digital assets.                                                     
───────────────────────────────┼────────────────────────────────────────────────────────────────────
Cloud Security `alpha`         │Scans cloud environments for misconfigurations and vulnerabilities. 
───────────────────────────────┼────────────────────────────────────────────────────────────────────
Mobile Application Security    │Scans mobile applications for security issues, including API tests  
                               │and configuration checks.                                           
───────────────────────────────┼────────────────────────────────────────────────────────────────────
Network Device Security `alpha`│Identifies vulnerabilities in network devices like routers,         
                               │switches, and firewalls.                                            
───────────────────────────────┼────────────────────────────────────────────────────────────────────
Web Server Assessment          │Identifies common vulnerabilities and misconfigurations in web      
                               │servers.                                                            
───────────────────────────────┼────────────────────────────────────────────────────────────────────
Content Management System (CMS)│Identifies vulnerabilities specific to CMS platforms like WordPress,
Assessment                     │Joomla, or Drupal.                                                  
───────────────────────────────┼────────────────────────────────────────────────────────────────────
Database Security Assessment   │Scans databases for known vulnerabilities, default configurations,  
                               │and access control issues.                                          
───────────────────────────────┴────────────────────────────────────────────────────────────────────

## [​][20]
## Who is Nuclei for?

People use Nuclei in a variety of ways:

* **Security Engineers/Analysts**: Conduct security assessments, proactively identify
  vulnerabilities, convert custom vectors and analyze latest attack vectors.
* **Red Teams**: Leverage Nuclei as part of their offensive security operations to simulate
  real-world attack scenarios, identify weaknesses, and provide actionable recommendations for
  enhancing overall security.
* **DevOps Teams**: Integrate Nuclei into their CI/CD pipelines to ensure continuous security and
  regression of custom vulnerabilities.
* **Bug Bounty Hunters**: Leverage Nuclei to find vulnerabilities across their programs listed on
  platforms like HackerOne, Bugcrowd, Intigriti etc.
* **Penetration Testers**: Utilize Nuclei to automate their assessment methodologies into templates
  for their clients’ systems.

### [​][21]
### Security Engineers

Nuclei offers a number of features that are helpful for security engineers to customise workflows in
their organization. With the varieties of scan capabilities (like DNS, HTTP, TCP), security
engineers can easily create a suite of custom checks with Nuclei.

* Protocols support including: TCP, DNS, HTTP, File, etc
* Achieve complex vulnerability steps with workflows and [dynamic requests.][22]
* Easily integrate into CI/CD, designed to be easily integrated into regression cycle to actively
  check the fix and re-appearance of vulnerability.

### [​][23]
### Developers and Organizations

Nuclei is built with simplicity in mind and templates backed by hundreds of community members, it
allows you to stay updated with the latest security threats using continuous Nuclei scanning on the
hosts. It is designed to be easily integrated into regression tests cycle, to verify fixes and
eliminate future vulnerabilities.

* **CI/CD:** Engineers are already using Nuclei within their CI/CD pipeline, it allows them to
  constantly monitor their staging and production environments with customised templates.
* **Continuous Regression Cycle:** With Nuclei, you can create your custom template on every new
  identified vulnerability and put into Nuclei engine to eliminate in the continuous regression
  cycle.

### [​][24]
### Bug Bounty Hunters

Nuclei allows a custom testing approach, supporting your own suite of checks to easily run across
your bug bounty programs. In addition, Nuclei can be easily integrated into any continuous scanning
workflow.

* Nuclei is easily integrated into other tool workflows
* Can process thousands of hosts in few minutes
* Easily automates your custom testing approach with our simple YAML DSL
Check our projects and tools to see what might fit into your bug bounty workflow:
[github.com/projectdiscovery][25]. We also host a daily [refresh of DNS data at Chaos][26].

### [​][27]
### Penetration Testers

Nuclei can immensely improve how you approach security assessment by augmenting the manual,
repetitive processes. Consultancies are already converting their manual assessment steps with
Nuclei, it allows them to run set of their custom assessment approach across thousands of hosts in
an automated manner. Pen-testers get the full power public templates and customization capabilities
to speed up their assessment process, particularly during the regression cycle where you can easily
verify the fix.

* Easily create your compliance, standards suite (e.g. OWASP Top 10) checklist
* Use capabilities like [DAST][28] and [workflows][29] to simplify complex manual steps and
  repetitive assessment through automated with Nuclei.
* Easy to re-test vulnerability-fix by just re-running the template.

[1]: #what-is-nuclei
[2]: /getstarted-overview
[3]: #what-are-nuclei’s-features
[4]: #
[5]: #
[6]: #
[7]: #
[8]: #
[9]: #
[10]: #
[11]: #
[12]: #
[13]: #
[14]: #
[15]: #
[16]: /opensource/nuclei/authenticated-scans
[17]: #
[18]: #
[19]: #how-can-i-use-nuclei
[20]: #who-is-nuclei-for
[21]: #security-engineers
[22]: https://blog.projectdiscovery.io/nuclei-unleashed-quickly-write-complex-exploits/
[23]: #developers-and-organizations
[24]: #bug-bounty-hunters
[25]: http://github.com/projectdiscovery
[26]: http://chaos.projectdiscovery.io
[27]: #penetration-testers
[28]: https://docs.projectdiscovery.io/templates/protocols/http/fuzzing-overview
[29]: https://docs.projectdiscovery.io/templates/workflows/overview
