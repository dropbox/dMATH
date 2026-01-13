# Nuclei Templates

#### Community curated list of templates for the nuclei engine to find security vulnerabilities in
#### applications.


[Documentation][1] • [Contributions][2] • [Discussion][3] • [Community][4] • [FAQs][5] • [Join
Discord][6]

Templates are the core of the [nuclei scanner][7] which powers the actual scanning engine. This
repository stores and houses various templates for the scanner provided by our team, as well as
contributed by the community. We hope that you also contribute by sending templates via **pull
requests** or [Github issues][8] to grow the list.

## Nuclei Templates overview

An overview of the nuclei template project, including statistics on unique tags, author, directory,
severity, and type of templates. The table below contains the top ten statistics for each matrix; an
expanded version of this is [available here][9], and also available in [JSON][10] format for
integration.

────────────────────────────────────────────────────────────────────────────────────────────────────
### 🚨 Known Exploited Vulnerabilities (KEV) Coverage                                               
                                                                                                    
Nuclei templates provide coverage for vulnerabilities actively exploited in the wild:               
                                                                                                    
───────────────┬─────────┬──────────────────────────────────────────────────────────────────────────
**KEV Source** │**Templat│**Description**                                                           
               │es**     │                                                                          
───────────────┼─────────┼──────────────────────────────────────────────────────────────────────────
🔴 **CISA KEV**│**454**  │[CISA Known Exploited Vulnerabilities Catalog][11]                        
───────────────┼─────────┼──────────────────────────────────────────────────────────────────────────
🟠 **VulnCheck │**1449** │[VulnCheck KEV][12] - Enhanced vulnerability                              
KEV**          │         │intelligence                                                              
───────────────┼─────────┼──────────────────────────────────────────────────────────────────────────
🟢 **Both      │**407**  │Templates covering vulnerabilities in both catalogs                       
Sources**      │         │                                                                          
───────────────┴─────────┴──────────────────────────────────────────────────────────────────────────
                                                                                                    
> 💡 **Total unique KEV templates: 1496** - Use `nuclei -tags kev,vkev` to scan for actively        
> exploited vulnerabilities                                                                         
                                                                                                    
## Nuclei Templates Top 10 statistics                                                               
                                                                                                    
─────────┬─────┬─────────────┬─────┬──────────┬─────┬────────┬─────┬────┬───────────────────────────
TAG      │COUNT│AUTHOR       │COUNT│DIRECTORY │COUNT│SEVERITY│COUNT│TYPE│COUNT                      
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
vuln     │6468 │dhiyaneshdk  │1894 │http      │9281 │info    │4353 │file│436                        
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
cve      │3587 │daffainfo    │905  │cloud     │659  │high    │2552 │dns │26                         
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
discovery│3265 │princechaddha│854  │file      │436  │medium  │2457 │    │                           
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
vkev     │1394 │dwisiswant0  │805  │network   │259  │critical│1555 │    │                           
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
panel    │1365 │ritikchaddha │678  │code      │251  │low     │330  │    │                           
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
xss      │1269 │pussycat0x   │675  │dast      │240  │unknown │54   │    │                           
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
wordpress│1261 │pikpikcu     │353  │workflows │205  │        │     │    │                           
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
exposure │1141 │pdteam       │314  │javascript│92   │        │     │    │                           
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
wp-plugin│1103 │pdresearch   │275  │ssl       │38   │        │     │    │                           
─────────┼─────┼─────────────┼─────┼──────────┼─────┼────────┼─────┼────┼───────────────────────────
osint    │848  │iamnoooob    │263  │dns       │23   │        │     │    │                           
─────────┴─────┴─────────────┴─────┴──────────┴─────┴────────┴─────┴────┴───────────────────────────
                                                                                                    
**873 directories, 11997 files**.                                                                   
────────────────────────────────────────────────────────────────────────────────────────────────────

## 📖 Documentation

Please navigate to [https://nuclei.projectdiscovery.io][13] for detailed documentation to **build**
new or your own **custom** templates. We have also added a set of templates to help you understand
how things work.

## 💪 Contributions

Nuclei-templates is powered by major contributions from the community. [Template contributions
][14], [Feature Requests][15] and [Bug Reports][16] are more than welcome.

[[Alt]][17]

## 💬 Discussion

Have questions / doubts / ideas to discuss? Feel free to open a discussion on [Github
discussions][18] board.

## 👨‍💻 Community

You are welcome to join the active [Discord Community][19] to discuss directly with project
maintainers and share things with others around security and automation. Additionally, you may
follow us on [Twitter][20] to be updated on all the things about Nuclei.


Thanks again for your contribution and keeping this community vibrant. ❤️

[1]: https://docs.projectdiscovery.io/templates/introduction
[2]: #-contributions
[3]: #-discussion
[4]: #-community
[5]: https://docs.projectdiscovery.io/templates/faq
[6]: https://discord.gg/projectdiscovery
[7]: https://github.com/projectdiscovery/nuclei
[8]: https://github.com/projectdiscovery/nuclei-templates/issues/new?assignees=&labels=&template=sub
mit-template.md&title=%5Bnuclei-template%5D+
[9]: /projectdiscovery/nuclei-templates/blob/main/TEMPLATES-STATS.md
[10]: /projectdiscovery/nuclei-templates/blob/main/TEMPLATES-STATS.json
[11]: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
[12]: https://vulncheck.com/kev
[13]: https://nuclei.projectdiscovery.io
[14]: https://github.com/projectdiscovery/nuclei-templates/issues/new?assignees=&labels=&template=su
bmit-template.md&title=%5Bnuclei-template%5D+
[15]: https://github.com/projectdiscovery/nuclei-templates/issues/new?assignees=&labels=&template=fe
ature_request.md&title=%5BFeature%5D+
[16]: https://github.com/projectdiscovery/nuclei-templates/issues/new?assignees=&labels=&template=bu
g_report.md&title=%5BBug%5D+
[17]: https://camo.githubusercontent.com/1287b679ed50ce971d57cb360c8ecdfc4222a2e09b8c1db9caa048ba2a1
87333/68747470733a2f2f7265706f62656174732e6178696f6d2e636f2f6170692f656d6265642f35356565363535343362
6239613066396337393736323663346536366434373261353137643137632e737667
[18]: https://github.com/projectdiscovery/nuclei-templates/discussions
[19]: https://discord.gg/projectdiscovery
[20]: https://twitter.com/pdnuclei
