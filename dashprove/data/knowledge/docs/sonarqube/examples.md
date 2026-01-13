# Sonar Scanning Examples

This repository showcases basic examples of usage and code coverage for SonarScanners.

* [SonarScanner for Gradle][1]
* [SonarScanner for .NET][2]
* [SonarScanner for Maven][3]
* SonarScanner CLI in a Java Ant project (Formerly [SonarScanner for Ant][4] - This scanner is now
  deprecated. See link for more details)
* [SonarScanner CLI][5]

Sonar's [Clean Code solution][6] helps developers deliver high-quality, efficient code standards
that benefit the entire team or organization.

## Examples

### Various Languages

* [SonarScanner for various languages][7]

### Ant

Scaning an Ant project is no different than scanning a plain Java (no build tool) project. Ant is
used to build the project, but not to run the scan. Instead, the SonarScanner CLI is used.

* [SonarScanner for Ant - Basic][8]
* [SonarScanner for Ant - Code Coverage][9]

[SonarScanner for Ant][10] is now deprecated. Please migrate to the SonarScanner CLI.

### Gradle

If you have a Gradle project, we recommend usage of [SonarScanner for Gradle][11] or the equivalent
SonarScanner for Gradle on your CI pipeline.

* [SonarScanner for Gradle - Basic][12]
* [SonarScanner for Gradle - Kotlin DSL][13]
* [SonarScanner for Gradle - Multi-Module][14]
* [SonarScanner for Gradle - Multi-Module Code Coverage][15]

### Maven

If you have a Maven project, we recommend the usage of [SonarScanner for Maven][16] or the
equivalent SonarScanner for Maven on your CI pipeline.

* [SonarScanner for Maven - Basic][17]
* [SonarScanner for Maven - Multilingual (Java + Kotlin with coverage)][18]
* [SonarScanner for Maven - Multi-Module][19]

### DotNet/C#

If you have a .NET project, we recommend the usage of [SonarScanner for .NET][20] or the equivalent
SonarScanner for .NET on your CI pipeline.

* [SonarScanner for .NET/MSBuild - C#][21]

### Swift

[SonarScanner - Swift Code Coverage][22]

### C/C++/Objective-C

***NOTE:*** All SonarScanner examples for C, C++ and Objective-C can be found [here][23].

## License

Copyright 2016-2025 SonarSource.

Licensed under the [GNU Lesser General Public License, Version 3.0][24]

[1]: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner-for-g
radle
[2]: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner-for-d
otnet
[3]: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner-for-m
aven
[4]: https://docs.sonarsource.com/sonarqube/9.9/analyzing-source-code/scanners/sonarscanner-for-ant
[5]: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner
[6]: https://www.sonarsource.com/solutions/clean-code/
[7]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner
[8]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-ant/ant-basic
[9]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-ant/ant-coverage
[10]: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner-for-
ant
[11]: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner-for-
gradle
[12]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-gradle/gradle-basic
[13]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-gradle/gradle-kotlin-dsl
[14]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-gradle/gradle-multimodule
[15]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-gradle/gradle-multimodule-cover
age
[16]: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner-for-
maven
[17]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-maven/maven-basic
[18]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-maven/maven-multilingual
[19]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-maven/maven-multimodule
[20]: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner-for-
dotnet
[21]: /SonarSource/sonar-scanning-examples/blob/master/sonar-scanner-msbuild/CSharpProject
[22]: /SonarSource/sonar-scanning-examples/blob/master/swift-coverage
[23]: https://github.com/sonarsource-cfamily-examples
[24]: http://www.gnu.org/licenses/lgpl.txt
