* [Intro][1]
* [News][2]
* [Release Notes][3]
* [Docs][4]
* [Who][5]

[[REST Assured]][6]

[[Build Status]][7] [[Maven Central]][8]

Testing and validating REST services in Java is harder than in dynamic languages such as Ruby and
Groovy. REST Assured brings the simplicity of using these languages into the Java domain. For
example, if your HTTP server returns the following [JSON][9] at “http://localhost:8080/lotto/{id}”:

`{
   "lotto":{
      "lottoId":5,
      "winning-numbers":[2,45,34,23,7,5,3],
      "winners":[
         {
            "winnerId":23,
            "numbers":[2,45,34,23,3,5]
         },
         {
            "winnerId":54,
            "numbers":[52,3,12,11,18,22]
         }
      ]
   }
}
`

You can easily use REST Assured to validate interesting things from the response:

`@Test public void
lotto_resource_returns_200_with_expected_id_and_winners() {

    when().
            get("/lotto/{id}", 5).
    then().
            statusCode(200).
            body("lotto.lottoId", equalTo(5),
                 "lotto.winners.winnerId", hasItems(23, 54));

}
`

Looks easy enough? Why not give it a spin? See [getting started][10] and [usage guide][11].

[ Fork me on GitHub ][12]

** ** News

## News

* 2025-12-12: REST Assured 6.0.0 is released. It raises the baseline to Java 17+, upgrades to Groovy
  5, adds Spring 7 + Jackson 3 support (and bumps minimum versions for Spring/Yasson/Johnzon). See
  [release notes][13] and [change log][14] for more details.
* 2025-08-15: REST Assured 5.5.6 is released with bug fixes and minor improvements. See [change
  log][15] for more details.
* 2025-05-22: REST Assured 5.5.5 is released that fixes problems with `rest-assured-bom` file after
  moving to new deployment method. See [change log][16] for more details.
** ** Release Notes

## Release Notes

[6.0.0][17]
[5.2.0][18]
[5.0.0][19]
[4.5.0][20]
[4.2.0][21]
[4.0.0][22]
[3.3.0][23]
[3.2.0][24]
[3.1.0][25]
[3.0.0][26]
[2.9.0][27]
[2.8.0][28]
[2.7.0][29]
[2.6.0][30]
[2.5.0][31]
[2.4.0][32]
[2.3.0][33]
[2.2.0][34]
[2.1.0][35]
[2.0.0][36]
[1.9.0][37]
[1.8.0][38]
[1.7][39]
[1.6][40]
[1.5][41]
[1.4.5][42]
[1.1][43]

For minor changes see [change log][44].

** ** Docs

## Documentation

* [Getting started][45]
* [Downloads][46]
* [Usage Guide][47] (click [here][48] for legacy documentation)
* [Javadoc][49]
* [Rest Assured Javadoc][50]
* [Rest AssuredMockMvc Javadoc][51]
* [XmlPath Javadoc][52]
* [JsonPath Javadoc][53]
* [Release Notes][54]
* [FAQ][55]
** ** Who

## Who

REST Assured is developed and maintained by [Johan Haleby][56] with the help of numerous other
[contributors][57] over the years. Would you like to contribute to the project in any way? Submit a
[pull request][58] or contact Johan at [Twitter][59].

Johan started the project when he was working at [Jayway][60] back in December of 2010. The project
is now sponsored by [Parkster][61].

REST Assured is [open source][62] — Issue [tracker][63] — Mailing [list][64]

[1]: #intro
[2]: #news
[3]: #release-notes
[4]: #docs
[5]: #who
[6]: https://github.com/rest-assured/rest-assured
[7]: https://github.com/rest-assured/rest-assured/actions/workflows/ci.yml
[8]: https://central.sonatype.com/artifact/io.rest-assured/rest-assured
[9]: http://www.json.org/
[10]: https://github.com/rest-assured/rest-assured/wiki/GettingStarted
[11]: https://github.com/rest-assured/rest-assured/wiki/Usage
[12]: https://github.com/rest-assured/rest-assured
[13]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes60
[14]: https://raw.githubusercontent.com/rest-assured/rest-assured/master/changelog.txt
[15]: https://raw.githubusercontent.com/rest-assured/rest-assured/master/changelog.txt
[16]: https://raw.githubusercontent.com/rest-assured/rest-assured/master/changelog.txt
[17]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes60
[18]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes52
[19]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes50
[20]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes45
[21]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes42
[22]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes40
[23]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes33
[24]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes32
[25]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes31
[26]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes30
[27]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes29
[28]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes28
[29]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes27
[30]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes26
[31]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes25
[32]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes24
[33]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes23
[34]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes22
[35]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes21
[36]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes20
[37]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes19
[38]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes18
[39]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes17
[40]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes16
[41]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes15
[42]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes145
[43]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes11
[44]: https://raw.githubusercontent.com/rest-assured/rest-assured/master/changelog.txt
[45]: https://github.com/rest-assured/rest-assured/wiki/GettingStarted
[46]: https://github.com/rest-assured/rest-assured/wiki/Downloads
[47]: https://github.com/rest-assured/rest-assured/wiki/Usage
[48]: https://github.com/rest-assured/rest-assured/wiki/Usage_Legacy
[49]: http://www.javadoc.io/doc/io.rest-assured/rest-assured/latest
[50]: http://static.javadoc.io/io.rest-assured/rest-assured/latest/io/restassured/RestAssured.html
[51]: http://static.javadoc.io/io.rest-assured/spring-mock-mvc/latest/io/restassured/module/mockmvc/
RestAssuredMockMvc.html
[52]: http://static.javadoc.io/io.rest-assured/xml-path/latest/io/restassured/path/xml/XmlPath.html
[53]: http://static.javadoc.io/io.rest-assured/json-path/latest/io/restassured/path/json/JsonPath.ht
ml
[54]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes
[55]: https://github.com/rest-assured/rest-assured/wiki/FAQ
[56]: https://twitter.com/johanhaleby
[57]: https://github.com/rest-assured/rest-assured/contributors
[58]: https://github.com/rest-assured/rest-assured
[59]: https://twitter.com/johanhaleby
[60]: https://www.jayway.com/
[61]: https://www.parkster.se
[62]: https://github.com/rest-assured/rest-assured
[63]: https://github.com/rest-assured/rest-assured/issues
[64]: https://groups.google.com/forum/#!forum/rest-assured
