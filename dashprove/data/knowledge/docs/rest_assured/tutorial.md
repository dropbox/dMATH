[ rest-assured ][1] / ** [rest-assured][2] ** Public

* ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][3].
* [ Notifications ][4] You must be signed in to change notification settings
* [ Fork 1.9k ][5]
* [ Star 7.1k ][6]

* [ Code ][7]
* [ Issues 566 ][8]
* [ Pull requests 22 ][9]
* [ Actions ][10]
* [ Projects 0 ][11]
* [ Wiki ][12]
* [ Security ][13]
  [
  
  ### Uh oh!
  
  
  ][14]
* [ Insights ][15]
Additional navigation options

* [ Code ][16]
* [ Issues ][17]
* [ Pull requests ][18]
* [ Actions ][19]
* [ Projects ][20]
* [ Wiki ][21]
* [ Security ][22]
* [ Insights ][23]

# GettingStarted

[Jump to bottom][24]
Johan Haleby edited this page Dec 12, 2025 · [83 revisions][25]

# Contents

1. [Maven / Gradle][26]
   
   1. [REST Assured][27]
   2. [JsonPath][28]
   3. [XmlPath][29]
   4. [JSON Schema Validation][30]
   5. [Spring Mock MVC][31]
   6. [Spring Web Test Client][32]
   7. [Scala Module][33]
   8. [Kotlin Extensions Module][34]
   9. [Java 9+][35]
2. [Non-maven users][36]
3. [Static Imports][37]
4. [Version 2.x][38]
5. [Documentation][39]

## Maven / Gradle Users

Add the following dependency to your pom.xml:

### REST Assured

Includes [JsonPath][40] and [XmlPath][41]

Maven:

<dependency>
      <groupId>io.rest-assured</groupId>
      <artifactId>rest-assured</artifactId>
      <version>6.0.0</version>
      <scope>test</scope>
</dependency>

Gradle:

testImplementation 'io.rest-assured:rest-assured:6.0.0'

Notes

1. You should place rest-assured before the JUnit dependency declaration in your pom.xml /
   build.gradle in order to make sure that the correct version of Hamcrest is used.
2. REST Assured includes JsonPath and XmlPath as transitive dependencies

### JsonPath

Standalone JsonPath (included if you depend on the `rest-assured` artifact). Makes it easy to parse
JSON documents. Note that this JsonPath implementation uses [Groovy's GPath][42] syntax and is not
to be confused with Kalle Stenflo's [JsonPath][43] implementation.

Maven:

<dependency>
      <groupId>io.rest-assured</groupId>
      <artifactId>json-path</artifactId>
      <version>6.0.0</version>
      <scope>test</scope>
</dependency>

Gradle:

testImplementation 'io.rest-assured:json-path:6.0.0'

### XmlPath

Stand-alone XmlPath (included if you depend on the `rest-assured` artifact). Makes it easy to parse
XML documents.

Maven:

<dependency>
      <groupId>io.rest-assured</groupId>
      <artifactId>xml-path</artifactId>
      <version>6.0.0</version>
      <scope>test</scope>
</dependency>

Gradle:

testImplementation 'io.rest-assured:xml-path:6.0.0'

### JSON Schema Validation

If you want to validate that a JSON response conforms to a [Json Schema][44] you can use the
`json-schema-validator` module:

Maven:

<dependency>
      <groupId>io.rest-assured</groupId>
      <artifactId>json-schema-validator</artifactId>
      <version>6.0.0</version>
      <scope>test</scope>
</dependency>

Gradle:

testImplementation 'io.rest-assured:json-schema-validator:6.0.0'

Refer to the [documentation][45] for more info.

### Spring Mock Mvc

If you're using Spring Mvc you can now unit test your controllers using the [RestAssuredMockMvc][46]
API in the [spring-mock-mvc][47] module. For this to work you need to depend on the
`spring-mock-mvc` module:

Maven:

<dependency>
      <groupId>io.rest-assured</groupId>
      <artifactId>spring-mock-mvc</artifactId>
      <version>6.0.0</version>
      <scope>test</scope>
</dependency>

Gradle:

testImplementation 'io.rest-assured:spring-mock-mvc:6.0.0'

Using Kotlin? Refer to the [documentation][48] for Kotlin extension functions that makes it nicer to
work with the [Spring Mock Mvc module][49].

### Spring Web Test Client

If you're using Spring Webflux you can now unit test your reactive controllers using the
[RestAssuredWebTestClient][50] API in the [spring-mock-mvc][51] module. For this to work you need to
depend on the `spring-web-test-client` module:

Maven:

<dependency>
      <groupId>io.rest-assured</groupId>
      <artifactId>spring-web-test-client</artifactId>
      <version>6.0.0</version>
      <scope>test</scope>
</dependency>

Gradle:

testImplementation 'io.rest-assured:spring-web-test-client:6.0.0'

### Scala Support

If you're using Scala you may leverage the [scala-support][52] module. For this to work you need to
depend on the `scala-support` module:

SBT:

libraryDependencies += "io.rest-assured" % "scala-support" % "6.0.0"

Maven:

<dependency>
    <groupId>io.rest-assured</groupId>
    <artifactId>scala-support</artifactId>
    <version>6.0.0</version>
    <scope>test</scope>
</dependency>

Gradle:

testImplementation 'io.rest-assured:scala-support:6.0.0'

### Kotlin

If you're using Kotlin then it's highly recommended to use the [Kotlin Extension Module][53]. This
modules provides some useful extension functions when working with REST Assured from Kotlin.

Maven:

<dependency>
    <groupId>io.rest-assured</groupId>
    <artifactId>kotlin-extensions</artifactId>
    <version>6.0.0</version>
    <scope>test</scope>
</dependency>

Gradle:

testImplementation 'io.rest-assured:kotlin-extensions:6.0.0'

Then import `Given` from the `io.restassured.module.kotlin.extensions` package.

If you're using the [Spring MockMvc module][54] please refer to the documentation [here][55] on how
to use custom Kotlin extension functions for this module.

### Java 9

When using Java 9+ and find yourself having problems with [split packages][56] you can depend on:

<dependency>
   <groupId>io.rest-assured</groupId>
   <artifactId>rest-assured-all</artifactId>
   <version>6.0.0</version>
   <scope>test</scope>
</dependency>

instead of just `rest-assured`.

## Non-maven users

Download [REST Assured][57] and [Json Schema Validator][58] (optional). You can also download
[XmlPath][59] and/or [JsonPath][60] separately if you don't need REST Assured. If you're using
Spring Mvc then you can download the [spring-mock-mvc][61] module as well. If you're using Spring
Web Test Client then you should download the [spring-web-test-client][62] module as well. If you're
using Scala you may optionally download the [scala-support][63] module. Kotlin users should download
the [kotlin-extensions][64] module. Extract the distribution zip file and put the jar files in your
class-path.

# Static imports

In order to use REST assured effectively it's recommended to statically import methods from the
following classes:

io.restassured.RestAssured.*
io.restassured.matcher.RestAssuredMatchers.*
org.hamcrest.Matchers.*

If you want to use [Json Schema][65] validation you should also statically import these methods:

io.restassured.module.jsv.JsonSchemaValidator.*

Refer to [Json Schema Validation][66] section for more info.

If you're using Spring MVC you can use the [spring-mock-mvc][67] module to unit test your Spring
Controllers using the Rest Assured DSL. To do this statically import the methods from
[RestAssuredMockMvc][68] *instead* of importing the methods from `io.rest-assured.RestAssured` and
`io.rest-assured.matcher.RestAssuredMatchers`:

io.restassured.module.mockmvc.RestAssuredMockMvc.*
io.restassured.matcher.RestAssuredMatchers.*

# Version 2.x

If you need to depend on an older version replace groupId `io.rest-assured` with
`com.jayway.restassured`.

# Documentation

When you've successfully downloaded and configured REST Assured in your classpath please refer to
the [usage guide][69] for examples.

## Toggle table of contents Pages 40

* Loading [
  Home
  ][70]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][71].
* Loading [
  Downloads
  ][72]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][73].
* Loading [
  FAQ
  ][74]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][75].
* Loading [
  GettingStarted
  ][76]
  
  * [Contents][77]
  * [Maven / Gradle Users][78]
  * [REST Assured][79]
  * [JsonPath][80]
  * [XmlPath][81]
  * [JSON Schema Validation][82]
  * [Spring Mock Mvc][83]
  * [Spring Web Test Client][84]
  * [Scala Support][85]
  * [Kotlin][86]
  * [Java 9][87]
  * [Non-maven users][88]
  * [Static imports][89]
  * [Version 2.x][90]
  * [Documentation][91]
* Loading [
  How_to_release
  ][92]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][93].
* Loading [
  Kotlin
  ][94]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][95].
* Loading [
  OldNews
  ][96]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][97].
* Loading [
  ReleaseNotes
  ][98]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][99].
* Loading [
  ReleaseNotes11
  ][100]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][101].
* Loading [
  ReleaseNotes145
  ][102]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][103].
* Loading [
  ReleaseNotes15
  ][104]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][105].
* Loading [
  ReleaseNotes16
  ][106]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][107].
* Loading [
  ReleaseNotes17
  ][108]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][109].
* Loading [
  ReleaseNotes18
  ][110]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][111].
* Loading [
  ReleaseNotes19
  ][112]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][113].
* Loading [
  ReleaseNotes20
  ][114]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][115].
* Loading [
  ReleaseNotes21
  ][116]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][117].
* Loading [
  ReleaseNotes22
  ][118]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][119].
* Loading [
  ReleaseNotes23
  ][120]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][121].
* Loading [
  ReleaseNotes24
  ][122]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][123].
* Loading [
  ReleaseNotes25
  ][124]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][125].
* Loading [
  ReleaseNotes26
  ][126]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][127].
* Loading [
  ReleaseNotes27
  ][128]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][129].
* Loading [
  ReleaseNotes28
  ][130]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][131].
* Loading [
  ReleaseNotes29
  ][132]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][133].
* Loading [
  ReleaseNotes30
  ][134]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][135].
* Loading [
  ReleaseNotes31
  ][136]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][137].
* Loading [
  ReleaseNotes32
  ][138]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][139].
* Loading [
  ReleaseNotes33
  ][140]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][141].
* Loading [
  ReleaseNotes40
  ][142]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][143].
* Loading [
  ReleaseNotes42
  ][144]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][145].
* Loading [
  ReleaseNotes45
  ][146]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][147].
* Loading [
  ReleaseNotes50
  ][148]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][149].
* Loading [
  ReleaseNotes52
  ][150]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][151].
* Loading [
  ReleaseNotes60
  ][152]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][153].
* Loading [
  Scala
  ][154]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][155].
* Loading [
  snapshot
  ][156]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][157].
* Loading [
  Spring
  ][158]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][159].
* Loading [
  Usage
  ][160]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][161].
* Loading [
  Usage_Legacy
  ][162]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][163].
* Show 25 more pages…

* [Getting Started][164]
* [Downloads][165]
* [Usage Guide][166] ([Legacy][167])
* [Snapshot dependencies][168]
* [Release Notes][169]
* [FAQ][170]
* [Support][171]

### Clone this wiki locally

[1]: /rest-assured
[2]: /rest-assured/rest-assured
[3]: 
[4]: /login?return_to=%2Frest-assured%2Frest-assured
[5]: /login?return_to=%2Frest-assured%2Frest-assured
[6]: /login?return_to=%2Frest-assured%2Frest-assured
[7]: /rest-assured/rest-assured
[8]: /rest-assured/rest-assured/issues
[9]: /rest-assured/rest-assured/pulls
[10]: /rest-assured/rest-assured/actions
[11]: /rest-assured/rest-assured/projects
[12]: /rest-assured/rest-assured/wiki
[13]: /rest-assured/rest-assured/security
[14]: /rest-assured/rest-assured/security
[15]: /rest-assured/rest-assured/pulse
[16]: /rest-assured/rest-assured
[17]: /rest-assured/rest-assured/issues
[18]: /rest-assured/rest-assured/pulls
[19]: /rest-assured/rest-assured/actions
[20]: /rest-assured/rest-assured/projects
[21]: /rest-assured/rest-assured/wiki
[22]: /rest-assured/rest-assured/security
[23]: /rest-assured/rest-assured/pulse
[24]: #wiki-pages-box
[25]: /rest-assured/rest-assured/wiki/GettingStarted/_history
[26]: #maven--gradle-users
[27]: #rest-assured
[28]: #jsonpath
[29]: #xmlpath
[30]: #json-schema-validation
[31]: #spring-mock-mvc
[32]: #spring-web-test-client
[33]: #scala-support
[34]: #kotlin
[35]: #java-9
[36]: #non-maven-users
[37]: #static-imports
[38]: #version-2x
[39]: #documentation
[40]: #jsonpath
[41]: #xmlpath
[42]: http://groovy-lang.org/processing-xml.html#_gpath
[43]: https://github.com/json-path/JsonPath
[44]: http://json-schema.org/
[45]: Usage#json-schema-validation
[46]: http://static.javadoc.io/io.restassured/spring-mock-mvc/6.0.0/io/restassured/module/mockmvc/Re
stAssuredMockMvc.html
[47]: https://github.com/jayway/rest-assured/wiki/Usage#spring-mock-mvc-module
[48]: https://github.com/rest-assured/rest-assured/wiki/Usage#kotlin-extension-module-for-spring-moc
kmvc
[49]: https://github.com/rest-assured/rest-assured/wiki/Usage#spring-mock-mvc-module
[50]: http://static.javadoc.io/io.restassured/spring-web-test-client/6.0.0/io/restassured/module/web
testclient/RestAssuredWebTestClient.html
[51]: https://github.com/rest-assured/rest-assured/wiki/Usage#spring-mock-mvc-module
[52]: https://github.com/jayway/rest-assured/wiki/Usage#scala-support-module
[53]: https://github.com/rest-assured/rest-assured/wiki/Usage#kotlin-extension-module
[54]: https://github.com/rest-assured/rest-assured/wiki/Usage#spring-mock-mvc-module
[55]: https://github.com/rest-assured/rest-assured/wiki/Usage#kotlin-extension-module-for-spring-moc
kmvc
[56]: https://www.logicbig.com/tutorials/core-java-tutorial/modules/split-packages.html
[57]: http://dl.bintray.com/johanhaleby/generic/rest-assured-6.0.0-dist.zip
[58]: http://dl.bintray.com/johanhaleby/generic/json-schema-validator-6.0.0-dist.zip
[59]: http://dl.bintray.com/johanhaleby/generic/xml-path-6.0.0-dist.zip
[60]: http://dl.bintray.com/johanhaleby/generic/json-path-6.0.0-dist.zip
[61]: http://dl.bintray.com/johanhaleby/generic/spring-mock-mvc-6.0.0-dist.zip
[62]: http://dl.bintray.com/johanhaleby/generic/spring-web-test-client-6.0.0-dist.zip
[63]: http://dl.bintray.com/johanhaleby/generic/scala-support-6.0.0-dist.zip
[64]: http://dl.bintray.com/johanhaleby/generic/kotlin-extensions-6.0.0-dist.zip
[65]: http://json-schema.org/
[66]: #json-schema-validation
[67]: https://github.com/rest-assured/rest-assured/wiki/Usage#spring-mock-mvc-module
[68]: http://static.javadoc.io/io.rest-assured/spring-mock-mvc/6.0.0/com/jayway/restassured/module/m
ockmvc/RestAssuredMockMvc.html
[69]: Usage
[70]: /rest-assured/rest-assured/wiki
[71]: 
[72]: /rest-assured/rest-assured/wiki/Downloads
[73]: 
[74]: /rest-assured/rest-assured/wiki/FAQ
[75]: 
[76]: /rest-assured/rest-assured/wiki/GettingStarted
[77]: /rest-assured/rest-assured/wiki/GettingStarted#contents
[78]: /rest-assured/rest-assured/wiki/GettingStarted#maven--gradle-users
[79]: /rest-assured/rest-assured/wiki/GettingStarted#rest-assured
[80]: /rest-assured/rest-assured/wiki/GettingStarted#jsonpath
[81]: /rest-assured/rest-assured/wiki/GettingStarted#xmlpath
[82]: /rest-assured/rest-assured/wiki/GettingStarted#json-schema-validation
[83]: /rest-assured/rest-assured/wiki/GettingStarted#spring-mock-mvc
[84]: /rest-assured/rest-assured/wiki/GettingStarted#spring-web-test-client
[85]: /rest-assured/rest-assured/wiki/GettingStarted#scala-support
[86]: /rest-assured/rest-assured/wiki/GettingStarted#kotlin
[87]: /rest-assured/rest-assured/wiki/GettingStarted#java-9
[88]: /rest-assured/rest-assured/wiki/GettingStarted#non-maven-users
[89]: /rest-assured/rest-assured/wiki/GettingStarted#static-imports
[90]: /rest-assured/rest-assured/wiki/GettingStarted#version-2x
[91]: /rest-assured/rest-assured/wiki/GettingStarted#documentation
[92]: /rest-assured/rest-assured/wiki/How_to_release
[93]: 
[94]: /rest-assured/rest-assured/wiki/Kotlin
[95]: 
[96]: /rest-assured/rest-assured/wiki/OldNews
[97]: 
[98]: /rest-assured/rest-assured/wiki/ReleaseNotes
[99]: 
[100]: /rest-assured/rest-assured/wiki/ReleaseNotes11
[101]: 
[102]: /rest-assured/rest-assured/wiki/ReleaseNotes145
[103]: 
[104]: /rest-assured/rest-assured/wiki/ReleaseNotes15
[105]: 
[106]: /rest-assured/rest-assured/wiki/ReleaseNotes16
[107]: 
[108]: /rest-assured/rest-assured/wiki/ReleaseNotes17
[109]: 
[110]: /rest-assured/rest-assured/wiki/ReleaseNotes18
[111]: 
[112]: /rest-assured/rest-assured/wiki/ReleaseNotes19
[113]: 
[114]: /rest-assured/rest-assured/wiki/ReleaseNotes20
[115]: 
[116]: /rest-assured/rest-assured/wiki/ReleaseNotes21
[117]: 
[118]: /rest-assured/rest-assured/wiki/ReleaseNotes22
[119]: 
[120]: /rest-assured/rest-assured/wiki/ReleaseNotes23
[121]: 
[122]: /rest-assured/rest-assured/wiki/ReleaseNotes24
[123]: 
[124]: /rest-assured/rest-assured/wiki/ReleaseNotes25
[125]: 
[126]: /rest-assured/rest-assured/wiki/ReleaseNotes26
[127]: 
[128]: /rest-assured/rest-assured/wiki/ReleaseNotes27
[129]: 
[130]: /rest-assured/rest-assured/wiki/ReleaseNotes28
[131]: 
[132]: /rest-assured/rest-assured/wiki/ReleaseNotes29
[133]: 
[134]: /rest-assured/rest-assured/wiki/ReleaseNotes30
[135]: 
[136]: /rest-assured/rest-assured/wiki/ReleaseNotes31
[137]: 
[138]: /rest-assured/rest-assured/wiki/ReleaseNotes32
[139]: 
[140]: /rest-assured/rest-assured/wiki/ReleaseNotes33
[141]: 
[142]: /rest-assured/rest-assured/wiki/ReleaseNotes40
[143]: 
[144]: /rest-assured/rest-assured/wiki/ReleaseNotes42
[145]: 
[146]: /rest-assured/rest-assured/wiki/ReleaseNotes45
[147]: 
[148]: /rest-assured/rest-assured/wiki/ReleaseNotes50
[149]: 
[150]: /rest-assured/rest-assured/wiki/ReleaseNotes52
[151]: 
[152]: /rest-assured/rest-assured/wiki/ReleaseNotes60
[153]: 
[154]: /rest-assured/rest-assured/wiki/Scala
[155]: 
[156]: /rest-assured/rest-assured/wiki/snapshot
[157]: 
[158]: /rest-assured/rest-assured/wiki/Spring
[159]: 
[160]: /rest-assured/rest-assured/wiki/Usage
[161]: 
[162]: /rest-assured/rest-assured/wiki/Usage_Legacy
[163]: 
[164]: GettingStarted
[165]: Downloads
[166]: Usage
[167]: Usage_Legacy
[168]: snapshot
[169]: ReleaseNotes
[170]: FAQ
[171]: http://groups.google.com/group/rest-assured
