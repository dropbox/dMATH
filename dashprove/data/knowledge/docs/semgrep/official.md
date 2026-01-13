* Docs home
[[Semgrep themed logo][Semgrep themed logo]][1]

# Semgrep docs

Find bugs and reachable dependency vulnerabilities in code. Enforce your code standards on every
commit.

### Scan with Semgrep AppSec Platform

Deploy static application security testing (SAST), software composition analysis (SCA), and secrets
scans from one platform.

[
Get started

Run your first Semgrep scan.

][2][
Deploy Semgrep

Deploy Semgrep to your organization quickly and at scale.

][3][
Triage and remediate

Triage and remediate findings; fine-tune guardrails for developers.

][4][
Write rules

Enforce your organization’s coding standards with custom rules.

][5]

### Supported languages

─────┬──────────────────────────────────────────────────────────────────────────────────────────────
Produ│Languages                                                                                     
ct   │                                                                                              
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
Semgr│**Generally available (GA)**                                                                  
ep   │C and C++ • C# • Generic • Go • Java • JavaScript • JSON • Kotlin • Python • TypeScript • Ruby
Code │• Rust • JSX • PHP • Scala • Swift • Terraform                                                
     │                                                                                              
     │**Beta**                                                                                      
     │APEX • Elixir                                                                                 
     │                                                                                              
     │**Experimental**                                                                              
     │Bash • Cairo • Circom • Clojure • Dart • Dockerfile • Hack • HTML • Jsonnet • Julia • Lisp •  
     │Lua • Move on Aptos • Move on Sui • OCaml• R • Scheme • Solidity • YAML • XML                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
Semgr│**Generally available reachability**                                                          
ep   │C# • Go • Java • JavaScript and TypeScript • Kotlin • PHP • Python • Ruby • Scala • Swift     
Suppl│                                                                                              
y    │**Languages without support for reachability analysis**                                       
Chain│Dart • Elixir • Rust                                                                          
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
Semgr│Language-agnostic; can detect 630+ types of credentials or keys.                              
ep   │                                                                                              
Secre│                                                                                              
ts   │                                                                                              
─────┴──────────────────────────────────────────────────────────────────────────────────────────────

See the [Supported languages][6] documentation for more details.

### November 2025 release notes summary

* **Cortex** and **Sysdig** integrations are now generally available. Semgrep now uses deployment
  status and, for Cortex, internet-exposure data from these CNAPP providers to better prioritize
  findings.
* Malicious dependency detection is now generally available. Semgrep detects malicious packages,
  including malware, typosquatting, and credential-stealing dependencies, using over 80,000 rules.
* Assistant now automatically analyzes **all new Critical and High-severity findings** with **Medium
  or High confidence** in full scans, removing the previous 10-issue limit.
* The **Settings > General** tab now displays all Semgrep product settings on a single page.

[See the latest release notes **][7]

[** Subscribe to RSS feed ][8]

Not finding what you need in this doc? Ask questions in our [Community Slack group][9], or see
[Support][10] for other ways to get help.

[Edit this page][11]
Last updated on Dec 9, 2025

[1]: https://semgrep.dev
[2]: /docs/getting-started/quickstart-managed-scans
[3]: /docs/deployment/core-deployment
[4]: /docs/semgrep-code/triage-remediation
[5]: /docs/writing-rules/overview
[6]: /docs/supported-languages#language-maturity-summary
[7]: /docs/release-notes
[8]: https://semgrep.dev/docs/release-notes/rss.xml
[9]: https://go.semgrep.dev/slack
[10]: /docs/support/
[11]: https://github.com/semgrep/semgrep-docs/edit/main/docs/index.md
