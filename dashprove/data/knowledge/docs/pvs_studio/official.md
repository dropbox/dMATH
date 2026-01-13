Dec 04 2025

# PVS-Studio manual

Dec 04 2025

* [Introduction][1]
* [Analyzing projects][2]
  
  * [On Windows][3]
  * [On Linux and macOS][4]
  * [Cross-platform][5]
  * [IDE][6]
  * [Build systems][7]
  * [Game Engines][8]
* [Continuous use of the analyzer in software development][9]
* [Integrating of PVS-Studio analysis results in code quality services (web dashboard)][10]
* [Deploying the analyzer in cloud Continuous Integration services][11]
* [Managing analysis results][12]
* [Additional configuration and resolving issues][13]
* [Analyzer diagnostics][14]
* [Additional information][15]

You can open full PVS-Studio documentation as [single page][16].

## Introduction

* [How to enter the PVS-Studio license and what's the next move][17]
* [PVS-Studio's trial mode][18]
* [System requirements][19]
* [Technologies used in PVS-Studio][20]
* [Release history][21]
* [Release history for previous versions (before 7.00)][22]

## Analyzing projects

### On Windows

* [Getting acquainted with the PVS-Studio static code analyzer on Windows][23]
* [Build-system independent analysis (C and C++)][24]
* [Direct integration of the analyzer into build automation systems (C and C++)][25]

### On Linux and macOS

* [PVS-Studio C# installation on Linux and macOS][26]
* [How to run PVS-Studio C# on Linux and macOS][27]
* [Installing and updating PVS-Studio C++ on Linux][28]
* [Installing and updating PVS-Studio C++ on macOS][29]
* [How to run PVS-Studio C++ on Linux and macOS][30]

### Cross-platform

* [Direct use of Java analyzer from command line][31]
* [Cross-platform analysis of C and C++ projects in PVS-Studio][32]
* [PVS-Studio for embedded development][33]
* [Analysis of C and C++ projects based on JSON Compilation Database][34]

### IDE

* [Get started with PVS-Studio in Visual Studio][35]
* [Using PVS-Studio with JetBrains Rider and CLion][36]
* [How to use the PVS-Studio extension for Qt Creator][37]
* [How to integrate PVS-Studio in Qt Creator without the PVS-Studio plugin][38]
* [Using the PVS-Studio extension for Visual Studio Code][39]
* [Using PVS-Studio with IntelliJ IDEA and Android Studio][40]

### Build systems

* [Analyzing Visual Studio / MSBuild / .NET projects from the command line using PVS-Studio][41]
* [Using PVS-Studio with the CMake module][42]
* [Integrating PVS-Studio Java into the Gradle build system][43]
* [Integrating PVS-Studio Java into the Maven build system][44]

### Game Engines

* [How to analyze Unity projects with PVS-Studio][45]
* [Analysis of Unreal Engine projects][46]

## Continuous use of the analyzer in software development

* [Running PVS-Studio in Docker][47]
* [Running PVS-Studio in Jenkins][48]
* [Running PVS-Studio in TeamCity][49]
* [How to upload analysis results to Jira][50]
* [PVS-Studio and continuous integration][51]
* [PVS-Studio's incremental analysis mode][52]
* [Analyzing commits and pull requests][53]
* [Unattended deployment of PVS-Studio][54]
* [Speeding up the analysis of C and C++ code through distributed build systems (Incredibuild)][55]

## Integrating of PVS-Studio analysis results in code quality services (web dashboard)

* [Integration of PVS-Studio analysis results into DefectDojo][56]
* [Integration of PVS-Studio analysis results into SonarQube][57]
* [Integration of PVS-Studio analysis results into CodeChecker][58]

## Deploying the analyzer in cloud Continuous Integration services

* [Using with Travis CI][59]
* [Using with CircleCI][60]
* [Using with GitLab CI/CD][61]
* [Using with GitHub Actions][62]
* [Using with Azure DevOps][63]
* [Using with AppVeyor][64]
* [Using with Buddy][65]

## Managing analysis results

* [How to display the analyzer's most interesting warnings][66]
* [How to use the OWASP diagnostic group in PVS-Studio][67]
* [MISRA Coding Standards and Compliance][68]
* [Baselining analysis results (suppressing warnings for existing code)][69]
* [Handling the diagnostic messages list in Visual Studio][70]
* [Suppression of false-positive warnings][71]
* [How to view and convert analyzer's results][72]
* [Relative paths in PVS-Studio log files][73]
* [Viewing analysis results with C and C++ Compiler Monitoring UI][74]
* [Notifying the developer teams (blame-notifier utility)][75]
* [Filtering and handling the analyzer output through diagnostic configuration files
  (.pvsconfig)][76]
* [Excluding files and directories from analysis][77]

## Additional configuration and resolving issues

* [Tips on speeding up PVS-Studio][78]
* [PVS-Studio: troubleshooting][79]
* [Additional diagnostics configuration][80]
* [User annotation mechanism in JSON format][81]
  
  * [Annotating C and C++ entities in JSON format][82]
  * [Annotating C# entities in JSON format][83]
  * [Annotating Java entities in JSON format][84]
* [Predefined PVS_STUDIO macro][85]
* [Analysis configuration file (Settings.xml)][86]
* PVS-Studio settings in Visual Studio / C and C++ compiler monitoring UI
  
  * [Settings: general][87]
  * [Settings: Common Analyzer Settings][88]
  * [Settings: Detectable Errors][89]
  * [Settings: Don't Check Files][90]
  * [Settings: Keyword Message Filtering][91]
  * [Settings: Registration][92]
  * [Settings: Specific Analyzer Settings][93]

## Analyzer diagnostics

* [PVS-Studio Messages][94]
  
  * [General Analysis (C++)][95]
  * [General Analysis (C#)][96]
  * [General Analysis (Java)][97]
  * [Micro-Optimizations (C++)][98]
  * [Micro-Optimizations (C#)][99]
  * [Diagnosis of 64-bit errors (Viva64, C++)][100]
  * [Customer specific requests (C++)][101]
  * [MISRA errors][102]
  * [AUTOSAR errors][103]
  * [OWASP errors (C++)][104]
  * [OWASP errors (C#)][105]
  * [OWASP errors (Java)][106]
  * [Problems related to code analyzer][107]

## Additional information

* [Credits and acknowledgements][108]

Was this page helpful?

**Message submitted. ** [check circle]

Your message has been sent. We will email you at

If you do not see the email in your inbox, please check if it is filtered to one of the following
folders:

* Promotion
* Updates
* Spam

If you can’t find an answer to your question, fill in the form below and our developers will contact
you

By clicking this button you agree to our [Privacy Policy][109] statement

[1]: #ID0B79795D3E
[2]: #ID806DA4C1AF
[3]: #ID2C86D8E9F3
[4]: #ID0EB51357F9
[5]: #ID07C7EEF9C1
[6]: #ID581D6381F3
[7]: #IDBBB7F39D08
[8]: #ID3B8887201D
[9]: #ID07E414F700
[10]: #ID0634466DF2
[11]: #ID4CE6FB2643
[12]: #ID8D522EB6CB
[13]: #IDAA7913AFE0
[14]: #ID1A66E89DFC
[15]: #ID0F68B904E3
[16]: /en/docs/manual/full/
[17]: /en/docs/manual/0046/
[18]: /en/docs/manual/0009/
[19]: /en/docs/manual/0018/
[20]: /en/docs/manual/6521/
[21]: /en/docs/manual/0010/
[22]: /en/docs/manual/0022/
[23]: /en/docs/manual/0007/
[24]: /en/docs/manual/0031/
[25]: /en/docs/manual/0006/
[26]: /en/docs/manual/0051/
[27]: /en/docs/manual/0035/
[28]: /en/docs/manual/0039/
[29]: /en/docs/manual/0042/
[30]: /en/docs/manual/0036/
[31]: /en/docs/manual/6703/
[32]: /en/docs/manual/6615/
[33]: /en/docs/manual/0045/
[34]: /en/docs/manual/6557/
[35]: /en/docs/manual/6522/
[36]: /en/docs/manual/0052/
[37]: /en/docs/manual/6648/
[38]: /en/docs/manual/6455/
[39]: /en/docs/manual/6646/
[40]: /en/docs/manual/6704/
[41]: /en/docs/manual/0035/
[42]: /en/docs/manual/6591/
[43]: /en/docs/manual/6706/
[44]: /en/docs/manual/6705/
[45]: /en/docs/manual/6607/
[46]: /en/docs/manual/0043/
[47]: /en/docs/manual/0047/
[48]: /en/docs/manual/0048/
[49]: /en/docs/manual/0049/
[50]: /en/docs/manual/6495/
[51]: /en/docs/manual/0005/
[52]: /en/docs/manual/0024/
[53]: /en/docs/manual/0055/
[54]: /en/docs/manual/0025/
[55]: /en/docs/manual/0041/
[56]: /en/docs/manual/6686/
[57]: /en/docs/manual/0037/
[58]: /en/docs/manual/6819/
[59]: /en/docs/manual/0057/
[60]: /en/docs/manual/0054/
[61]: /en/docs/manual/0056/
[62]: /en/docs/manual/6579/
[63]: /en/docs/manual/0053/
[64]: /en/docs/manual/6667/
[65]: /en/docs/manual/6668/
[66]: /en/docs/manual/6532/
[67]: /en/docs/manual/6536/
[68]: /en/docs/manual/6966/
[69]: /en/docs/manual/0032/
[70]: /en/docs/manual/0021/
[71]: /en/docs/manual/0017/
[72]: /en/docs/manual/0038/
[73]: /en/docs/manual/0027/
[74]: /en/docs/manual/0033/
[75]: /en/docs/manual/0050/
[76]: /en/docs/manual/6630/
[77]: /en/docs/manual/6640/
[78]: /en/docs/manual/0023/
[79]: /en/docs/manual/0029/
[80]: /en/docs/manual/0040/
[81]: /en/docs/manual/6810/
[82]: /en/docs/manual/6743/
[83]: /en/docs/manual/6808/
[84]: /en/docs/manual/7180/
[85]: /en/docs/manual/0020/
[86]: /en/docs/manual/6653/
[87]: /en/docs/manual/0011/
[88]: /en/docs/manual/0012/
[89]: /en/docs/manual/0013/
[90]: /en/docs/manual/0014/
[91]: /en/docs/manual/0015/
[92]: /en/docs/manual/0016/
[93]: /en/docs/manual/0030/
[94]: /en/docs/warnings/
[95]: /en/docs/warnings/#GeneralAnalysisCPP
[96]: /en/docs/warnings/#GeneralAnalysisCS
[97]: /en/docs/warnings/#GeneralAnalysisJAVA
[98]: /en/docs/warnings/#MicroOptimizationsCPP
[99]: /en/docs/warnings/#MicroOptimizationsCS
[100]: /en/docs/warnings/#64CPP
[101]: /en/docs/warnings/#CustomersSpecificRequestsCPP
[102]: /en/docs/warnings/#MISRA
[103]: /en/docs/warnings/#AUTOSAR
[104]: /en/docs/warnings/#OWASPCPP
[105]: /en/docs/warnings/#OWASPCS
[106]: /en/docs/warnings/#OWASPJava
[107]: /en/docs/warnings/#ProblemsRelatedToCodeAnalyzer
[108]: /en/docs/manual/0002/
[109]: https://pvs-studio.com/en/privacy-policy/
