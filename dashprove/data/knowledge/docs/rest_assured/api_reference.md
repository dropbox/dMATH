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

# Usage

[Jump to bottom][24]
Johan Haleby edited this page Dec 12, 2025 · [242 revisions][25]

Note that if you're using version 1.9.0 or earlier please refer to the [legacy][26] documentation.

REST Assured is a Java DSL for simplifying testing of REST based services built on top of HTTP
Builder. It supports POST, GET, PUT, DELETE, OPTIONS, PATCH and HEAD requests and can be used to
validate and verify the response of these requests.

# Contents

1.  [Static imports][27]
2.  [Examples][28]
    
    1. [JSON Example][29]
    2. [JSON Schema Validation][30]
    3. [XML Example][31]
    4. [Advanced][32]
       
       1. [XML][33]
       2. [JSON][34]
    5. [De-serialization with Generics][35]
    6. [Additional Examples][36]
3.  [Note on floats and doubles][37]
4.  [Note on syntax][38] ([syntactic sugar][39])
5.  [Getting Response Data][40]
    
    1. [Extracting values from the Response after validation][41]
    2. [JSON (using JsonPath)][42]
    3. [XML (using XmlPath)][43]
    4. [Single Path][44]
    5. [Headers, cookies, status etc][45]
       
       1. [Multi-value headers][46]
       2. [Multi-value cookies][47]
       3. [Detailed Cookies][48]
6.  [Specifying Request Data][49]
    
    1. [Invoking HTTP resources][50]
    2. [Parameters][51]
    3. [Multi-value Parameter][52]
    4. [No-value Parameter][53]
    5. [Path Parameters][54]
    6. [Cookies][55]
    7. [Headers][56]
    8. [Content-Type][57]
    9. [Request Body][58]
7.  [Verifying Response Data][59]
    
    1. [Response Body][60]
    2. [Cookies][61]
    3. [Status][62]
    4. [Headers][63]
    5. [Content-Type][64]
    6. [Full body/content matching][65]
    7. [Use the response to verify other parts of the response][66]
    8. [Measuring response time][67]
8.  [Authentication][68]
    
    1. [Basic][69]
       
       1. [Preemptive][70]
       2. [Challenged][71]
    2. [Digest][72]
    3. [Form][73]
    4. [OAuth][74]
       
       1. [OAuth1][75]
       2. [OAuth2][76]
9.  [CSRF][77]
    
    1. [Form Token][78]
    2. [Header Token][79]
    3. [Authentication][80]
    4. [Prioritization][81]
    5. [Cookie Propagation][82]
10. [Multi-part form data][83]
11. [Object Mapping][84]
    
    1. [Jakarta EE][85]
    2. [JAXB][86]
    3. [Serialization][87]
       
       1. [Content-Type based Serialization][88]
       2. [Create JSON from a HashMap][89]
       3. [Using an Explicit Serializer][90]
    4. [Deserialization][91]
       
       1. [Content-Type based Deserialization][92]
       2. [Custom Content-Type Deserialization][93]
       3. [Using an Explicit Deserializer][94]
    5. [Configuration][95]
    6. [Custom][96]
12. Parsers
    
    1. [Custom][97]
    2. [Default][98]
13. [Default Values][99]
14. [Specification Re-use][100]
    
    1. [Querying RequestSpecification][101]
15. [Filters][102]
    
    1. [Ordered Filters][103]
    2. [Response Builder][104]
16. [Logging][105]
    
    1. [Request Logging][106]
    2. [Response Logging][107]
    3. [Log if validation fails][108]
    4. [Blacklist Headers from Logging][109]
17. [Root Path][110]
    
    1. [Path Arguments][111]
18. [Session Support][112]
    
    1. [Session Filter][113]
19. [SSL][114]
    
    1. [SSL invalid hostname][115]
20. [URL Encoding][116]
21. [Proxy Configuration][117]
    
    1. [Static Proxy Configuration][118]
    2. [Request Specification Proxy Configuration][119]
22. [Detailed configuration][120]
    
    1.  [Encoder Config][121]
    2.  [Decoder Config][122]
    3.  [Session Config][123]
    4.  [Redirect DSL][124]
    5.  [Connection Config][125]
    6.  [JSON Config][126]
    7.  [HTTP Client Config][127]
    8.  [SSL Config][128]
    9.  [Param Config][129]
    10. [Failure Config][130]
    11. [CSRF Config][131]
23. [Spring Support][132]
    
    1. [Spring Mock Mvc Module][133]
       
       1. [Bootstrapping RestAssuredMockMvc][134]
       2. [Asynchronous Requests][135]
       3. [Adding Request Post Processors][136]
       4. [Adding Result Handlers][137]
       5. [Using Result Matchers][138]
       6. [Interceptors][139]
       7. [Specifications][140]
       8. [Resetting RestAssuredMockMvc][141]
       9. [Spring MVC Authentication][142]
          
          1. [Using Spring Security Test][143]
          2. [Injecting a User][144]
    2. [Spring Web Test Client Module][145]
       
       1. [Bootstrapping RestAssuredWebTestClient][146]
       2. [Specifications][147]
       3. [Resetting RestAssuredWebTestClient][148]
       4. [Kotlin Extension Module for Spring WebTest][149]
    3. [Common Spring Module Documentation][150]
       
       1. [Note on parameters][151]
24. [Scala][152]
    
    1. [Scala Extension Module][153]
    2. [Support Module][154]
25. [Kotlin][155]
    
    1. [Avoid Escaping "when" Keyword][156]
    2. [Kotlin Extension Module][157]
    3. [Kotlin Extension Module for Spring MockMvc][158]
    4. [Kotlin Extension Module for Spring WebTest][159]
26. [More Info][160]

## Static imports

In order to use REST assured effectively it's recommended to statically import methods from the
following classes:

io.restassured.RestAssured.*
io.restassured.matcher.RestAssuredMatchers.*
org.hamcrest.Matchers.*

If you want to use [Json Schema][161] validation you should also statically import these methods:

io.restassured.module.jsv.JsonSchemaValidator.*

Refer to [Json Schema Validation][162] section for more info.

If you're using Spring MVC you can use the [spring-mock-mvc][163] module to unit test your Spring
Controllers using the Rest Assured DSL. To do this statically import the methods from
[RestAssuredMockMvc][164] *instead* of importing the methods from `io.restassured.RestAssured`:

io.restassured.module.mockmvc.RestAssuredMockMvc.*

# Examples

## Example 1 - JSON

Assume that the GET request (to [http://localhost:8080/lotto][165]) returns JSON as:

{
"lotto":{
 "lottoId":5,
 "winning-numbers":[2,45,34,23,7,5,3],
 "winners":[{
   "winnerId":23,
   "numbers":[2,45,34,23,3,5]
 },{
   "winnerId":54,
   "numbers":[52,3,12,11,18,22]
 }]
}
}

REST assured can then help you to easily make the GET request and verify the response. E.g. if you
want to verify that lottoId is equal to 5 you can do like this:

get("/lotto").then().body("lotto.lottoId", equalTo(5));

or perhaps you want to check that the winnerId's are 23 and 54:

get("/lotto").then().body("lotto.winners.winnerId", hasItems(23, 54));

Note: `equalTo` and `hasItems` are Hamcrest matchers which you should statically import from
`org.hamcrest.Matchers`.

Note that the "json path" syntax uses [Groovy's GPath][166] notation and is not to be confused with
Jayway's [JsonPath][167] syntax.

### Returning floats and doubles as BigDecimal

You can configure Rest Assured and JsonPath to return BigDecimal's instead of float and double for
Json Numbers. For example consider the following JSON document:

{

    "price":12.12 

}

By default you validate that price is equal to 12.12 as a float like this:

get("/price").then().body("price", is(12.12f));

but if you like you can configure REST Assured to use a JsonConfig that returns all Json numbers as
BigDecimal:

given().
        config(RestAssured.config().jsonConfig(jsonConfig().numberReturnType(BIG_DECIMAL))).
when().
        get("/price").
then().
        body("price", is(new BigDecimal(12.12));

### JSON Schema validation

From version `2.1.0` REST Assured has support for [Json Schema][168] validation. For example given
the following schema located in the classpath as `products-schema.json`:

{
    "$schema": "http://json-schema.org/draft-04/schema#",
    "title": "Product set",
    "type": "array",
    "items": {
        "title": "Product",
        "type": "object",
        "properties": {
            "id": {
                "description": "The unique identifier for a product",
                "type": "number"
            },
            "name": {
                "type": "string"
            },
            "price": {
                "type": "number",
                "minimum": 0,
                "exclusiveMinimum": true
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 1,
                "uniqueItems": true
            },
            "dimensions": {
                "type": "object",
                "properties": {
                    "length": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"}
                },
                "required": ["length", "width", "height"]
            },
            "warehouseLocation": {
                "description": "Coordinates of the warehouse with the product",
                "$ref": "http://json-schema.org/geo"
            }
        },
        "required": ["id", "name", "price"]
    }
}

you can validate that a resource (`/products`) conforms with the schema:

get("/products").then().assertThat().body(matchesJsonSchemaInClasspath("products-schema.json"));

`matchesJsonSchemaInClasspath` is statically imported from
`io.restassured.module.jsv.JsonSchemaValidator` and it's recommended to statically import all
methods from this class. However in order to use it you need to depend on the
`json-schema-validator` module by either [downloading][169] it from the download page or add the
following dependency from Maven:

<dependency>
    <groupId>io.rest-assured</groupId>
    <artifactId>json-schema-validator</artifactId>
    <version>6.0.0</version>
</dependency>

### JSON Schema Validation Settings

REST Assured's `json-schema-validator` module uses Francis Galiegue's [json-schema-validator][170]
(`fge`) library to perform validation. If you need to configure the underlying `fge` library you can
for example do like this:

// Given
JsonSchemaFactory jsonSchemaFactory = JsonSchemaFactory.newBuilder().setValidationConfiguration(Vali
dationConfiguration.newBuilder().setDefaultVersion(DRAFTV4).freeze()).freeze();

// When
get("/products").then().assertThat().body(matchesJsonSchemaInClasspath("products-schema.json").using
(jsonSchemaFactory));

The `using` method allows you to pass in a `jsonSchemaFactory` instance that REST Assured will use
during validation. This allows fine-grained configuration for the validation.

The `fge` library also allows the validation to be `checked` or `unchecked`. By default REST Assured
uses `checked` validation but if you want to change this you can supply an instance of
[JsonSchemaValidatorSettings][171] to the matcher. For example:

get("/products").then().assertThat().body(matchesJsonSchemaInClasspath("products-schema.json").using
(settings().with().checkedValidation(false)));

Where the `settings` method is statically imported from the [JsonSchemaValidatorSettings][172]
class.

### Json Schema Validation with static configuration

Now imagine that you always want to use `unchecked` validation as well as setting the default json
schema version to version 3. Instead of supplying this to all matchers throughout your code you can
define it statically. For example:

JsonSchemaValidator.settings = settings().with().jsonSchemaFactory(
        JsonSchemaFactory.newBuilder().setValidationConfiguration(ValidationConfiguration.newBuilder
().setDefaultVersion(DRAFTV3).freeze()).freeze()).
        and().with().checkedValidation(false);

get("/products").then().assertThat().body(matchesJsonSchemaInClasspath("products-schema.json"));

Now any `matcher` method imported from [JsonSchemaValidator][173] will use `DRAFTV3` as default
version and unchecked validation.

To reset the `JsonSchemaValidator` to its default settings simply call the `reset` method:

JsonSchemaValidator.reset();

### Json Schema Validation without REST Assured

You can also use the `json-schema-validator` module without depending on REST Assured. As long as
you have a JSON document represented as a `String` you can do like this:

import org.junit.Test;
import static io.restassured.module.jsv.JsonSchemaValidator.matchesJsonSchemaInClasspath;
import static org.hamcrest.MatcherAssert.assertThat;
 
public class JsonSchemaValidatorWithoutRestAssuredTest {
 
 
    @Test public void
    validates_schema_in_classpath() {
        // Given
        String json = ... // Greeting response
 
        // Then
        assertThat(json, matchesJsonSchemaInClasspath("greeting-schema.json"));
    }
}

Refer to the [getting started][174] page for more info on this.

### Anonymous JSON root validation

A JSON document doesn't necessarily need a named root attribute. This is for example valid JSON:

[1, 2, 3]

An anonymous JSON root can be verified by using `$` or an empty string as path. For example let's
say that this JSON document is exposed from `http://localhost:8080/json` then we can validate it
like this with REST Assured:

when().
     get("/json").
then().
     body("$", hasItems(1, 2, 3)); // An empty string "" would work as well

## Example 2 - XML

XML can be verified in a similar way. Imagine that a POST request to
`http://localhost:8080/greetXML` returns:

<greeting>
   <firstName>{params("firstName")}</firstName>
   <lastName>{params("lastName")}</lastName>
</greeting>

i.e. it sends back a greeting based on the firstName and lastName parameter sent in the request. You
can easily perform and verify e.g. the firstName with REST assured:

given().
         parameters("firstName", "John", "lastName", "Doe").
when().
         post("/greetXML").
then().
         body("greeting.firstName", equalTo("John")).

If you want to verify both firstName and lastName you may do like this:

given().
         parameters("firstName", "John", "lastName", "Doe").
when().
         post("/greetXML").
then().
         body("greeting.firstName", equalTo("John")).
         body("greeting.lastName", equalTo("Doe"));

or a little shorter:

with().parameters("firstName", "John", "lastName", "Doe").when().post("/greetXML").then().body("gree
ting.firstName", equalTo("John"), "greeting.lastName", equalTo("Doe"));

See [this][175] link for more info about the syntax (it follows Groovy's [GPath][176] syntax).

### XML namespaces

To make body expectations take namespaces into account you need to declare the namespaces using the
[io.restassured.config.XmlConfig][177]. For example let's say that a resource called
`namespace-example` located at `http://localhost:8080` returns the following XML:

<foo xmlns:ns="http://localhost/">
  <bar>sudo </bar>
  <ns:bar>make me a sandwich!</ns:bar>
</foo>

You can then declare the `http://localhost/` uri and validate the response:

given().
        config(RestAssured.config().xmlConfig(xmlConfig().declareNamespace("test", "http://localhost
/"))).
when().
         get("/namespace-example").
then().
         body("foo.bar.text()", equalTo("sudo make me a sandwich!")).
         body(":foo.:bar.text()", equalTo("sudo ")).
         body("foo.test:bar.text()", equalTo("make me a sandwich!"));

The path syntax follows Groovy's XmlSlurper syntax. Note that in versions prior to 2.6.0 the path
syntax was *not* following Groovy's XmlSlurper syntax. Please see [release notes][178] for versin
2.6.0 to see how the previous syntax looked like.

### XPath

You can also verify XML responses using x-path. For example:

given().parameters("firstName", "John", "lastName", "Doe").when().post("/greetXML").then().body(hasX
Path("/greeting/firstName", containsString("Jo")));

or

given().parameters("firstName", "John", "lastName", "Doe").post("/greetXML").then().body(hasXPath("/
greeting/firstName[text()='John']"));

To use namespaces in the XPath expression you need to enable them in the configuration, for example:

given().
        config(RestAssured.config().xmlConfig(xmlConfig().with().namespaceAware(true))).
when().
         get("/package-db-xml").
then().
         body(hasXPath("/db:package-database", namespaceContext));

Where `namespaceContext` is an instance of [javax.xml.namespace.NamespaceContext][179].

### Schema and DTD validation

XML response bodies can also be verified against an XML Schema (XSD) or DTD.

#### XSD example

get("/carRecords").then().assertThat().body(matchesXsd(xsd));

#### DTD example

get("/videos").then().assertThat().body(matchesDtd(dtd));

The `matchesXsd` and `matchesDtd` methods are Hamcrest matchers which you can import from
[io.restassured.matcher.RestAssuredMatchers][180].

## Example 3 - Complex parsing and validation

This is where REST Assured really starts to shine! Since REST Assured is implemented in Groovy it
can be really beneficial to take advantage of Groovy’s collection API. Let’s begin by looking at an
example in Groovy:

def words = ['ant', 'buffalo', 'cat', 'dinosaur']
def wordsWithSizeGreaterThanFour = words.findAll { it.length() > 4 }

At the first line we simply define a list with some words but the second line is more interesting.
Here we search the words list for all words that are longer than 4 characters by calling the findAll
with a Groovy closure. The closure has an implicit variable called `it` which represents the current
item in the list. The result is a new list, `wordsWithSizeGreaterThanFour`, containing `buffalo` and
`dinosaur`.

There are other interesting methods that we can use on collections in Groovy as well, for example:

* `find` – finds the first item matching a closure predicate
* `collect` – collect the return value of calling a closure on each item in a collection
* `sum` – Sum all the items in the collection
* `max`/`min` – returns the max/min values of the collection

So how do we take advantage of this when validating our XML or JSON responses with REST Assured?

### XML Example

Let’s say we have a resource at `http://localhost:8080/shopping` that returns the following XML:

<shopping>
      <category type="groceries">
        <item>Chocolate</item>
        <item>Coffee</item>
      </category>
      <category type="supplies">
        <item>Paper</item>
        <item quantity="4">Pens</item>
      </category>
      <category type="present">
        <item when="Aug 10">Kathryn's Birthday</item>
      </category>
</shopping>

Let’s also say we want to write a test that verifies that the category of type groceries has items
Chocolate and Coffee. In REST Assured it can look like this:

when().
       get("/shopping").
then().
       body("shopping.category.find { it.@type == 'groceries' }.item", hasItems("Chocolate", "Coffee
"));

What's going on here? First of all the XML path `shopping.category` returns a list of all
categories. On this list we invoke a function, `find`, to return the single category that has the
XML attribute, `type`, equal to `groceries`. On this category we then continue by getting all the
items associated with this category. Since there are more than one item associated with the
groceries category a list will be returned and we verify this list against the `hasItems` Hamcrest
matcher.

But what if you want to get the items and not validate them against a Hamcrest matcher? For this
purpose you can use [XmlPath][181]:

// Get the response body as a String
String response = get("/shopping").asString();
// And get the groceries from the response. "from" is statically imported from the XmlPath class
List<String> groceries = from(response).getList("shopping.category.find { it.@type == 'groceries' }.
item");

If the list of groceries is the only thing you care about in the response body you can also use a
[shortcut][182]:

// Get the response body as a String
List<String> groceries = get("/shopping").path("shopping.category.find { it.@type == 'groceries' }.i
tem");

#### Depth-first search

It's actually possible to simplify the previous example even further:

when().
       get("/shopping").
then().
       body("**.find { it.@type == 'groceries' }", hasItems("Chocolate", "Coffee"));

`**` is a shortcut for doing depth first searching in the XML document. We search for the first node
that has an attribute named `type` equal to "groceries". Notice also that we don't end the XML path
with "item". The reason is that `toString()` is called automatically on the category node which
returns a list of the item values.

### JSON Example

Let's say we have a resource at `http://localhost:8080/store` that returns the following JSON
document:

{  
   "store":{  
      "book":[  
         {  
            "author":"Nigel Rees",
            "category":"reference",
            "price":8.95,
            "title":"Sayings of the Century"
         },
         {  
            "author":"Evelyn Waugh",
            "category":"fiction",
            "price":12.99,
            "title":"Sword of Honour"
         },
         {  
            "author":"Herman Melville",
            "category":"fiction",
            "isbn":"0-6.0.01311-3",
            "price":8.99,
            "title":"Moby Dick"
         },
         {  
            "author":"J. R. R. Tolkien",
            "category":"fiction",
            "isbn":"0-395-19395-8",
            "price":22.99,
            "title":"The Lord of the Rings"
         }
      ]
   }
}

#### Example 1

As a first example let's say we want to make the request to "/store" and assert that the titles of
the books with a price less than 10 are "Sayings of the Century" and "Moby Dick":

when().
       get("/store").
then().
       body("store.book.findAll { it.price < 10 }.title", hasItems("Sayings of the Century", "Moby D
ick"));

Just as in the XML examples above we use a closure to find all books with a price less than 10 and
then return the titles of all the books. We then use the `hasItems` matcher to assert that the
titles are the ones we expect. Using [JsonPath][183] we can return the titles instead:

// Get the response body as a String
String response = get("/store").asString();
// And get all books with price < 10 from the response. "from" is statically imported from the JsonP
ath class
List<String> bookTitles = from(response).getList("store.book.findAll { it.price < 10 }.title");

#### Example 2

Let's consider instead that we want to assert that the sum of the length of all author names are
greater than 50. This is a rather complex question to answer and it really shows the strength of
closures and Groovy collections. In REST Assured it looks like this:

when().
       get("/store");
then().
       body("store.book.author.collect { it.length() }.sum()", greaterThan(50));

First we get all the authors (`store.book.author`) and invoke the collect method on the resulting
list with the closure `{ it.length() }`. What it does is to call the `length()` method on each
author in the list and returns the result to a new list. On this list we simply call the `sum()`
method to sum all the length's. The end result is `53` and we assert that it's greater than 50 by
using the `greaterThan` matcher. But it's actually possible to simplify this even further. Consider
the "[words][184]" example again:

def words = ['ant', 'buffalo', 'cat', 'dinosaur']

Groovy has a very handy way of calling a function for each element in the list by using the spread
operator, `*`. For example:

def words = ['ant', 'buffalo', 'cat', 'dinosaur']
assert [3, 6, 3, 8] == words*.length()

I.e. Groovy returns a new list with the lengths of the items in the words list. We can utilize this
for the author list in REST Assured as well:

when().
       get("/store");
then().
       body("store.book.author*.length().sum()", greaterThan(50)).

And of course we can use [JsonPath][185] to actually return the result:

// Get the response body as a string
String response = get("/store").asString();
// Get the sum of all author length's as an int. "from" is again statically imported from the JsonPa
th class
int sumOfAllAuthorLengths = from(response).getInt("store.book.author*.length().sum()");
// We can also assert that the sum is equal to 53 as expected.
assertThat(sumOfAllAuthorLengths, is(53));

## Deserialization with Generics

REST Assured 3.3.0 introduced the `io.restassured.mapper.TypeRef` class that allows you to
de-serialize the response to a container with a generic type. For example let's say that you have a
service that returns the following JSON for a GET request to `/products`:

[
          {
              "id": 2,
              "name": "An ice sculpture",
              "price": 12.50,
              "tags": ["cold", "ice"],
              "dimensions": {
                  "length": 7.0,
                  "width": 12.0,
                  "height": 9.5
              },
              "warehouseLocation": {
                  "latitude": -78.75,
                  "longitude": 20.4
              }
          },
          {
              "id": 3,
              "name": "A blue mouse",
              "price": 25.50,
                  "dimensions": {
                  "length": 3.1,
                  "width": 1.0,
                  "height": 1.0
              },
              "warehouseLocation": {
                  "latitude": 54.4,
                  "longitude": -32.7
              }
          }
      ]

You can then extract the root list to a `List<Map<String, Object>>` (or a any generic container of
choice) using the `TypeRef`:

// Extract
List<Map<String, Object>> products = get("/products").as(new TypeRef<List<Map<String, Object>>>() {}
);

// Now you can do validations on the extracted objects:
assertThat(products, hasSize(2));
assertThat(products.get(0).get("id"), equalTo(2));
assertThat(products.get(0).get("name"), equalTo("An ice sculpture"));
assertThat(products.get(0).get("price"), equalTo(12.5));
assertThat(products.get(1).get("id"), equalTo(3));
assertThat(products.get(1).get("name"), equalTo("A blue mouse"));
assertThat(products.get(1).get("price"), equalTo(25.5));```

Note that currently this only works for JSON responses.

## Additional Examples

Micha Kops has written a really good blog with several examples (including code examples that you
can checkout). You can read it [here][186].

Also [Bas Dijkstra][187] has been generous enough to open source his REST Assured workshop. You can
read more about this [here][188] and you can try out, and contribute to, the exercises available in
[his][189] github repository.

Bas has also made a nice introductory screencast to REST Assured, you can find it [here][190].

## Note on floats and doubles

Floating point numbers must be compared with a Java "float" primitive. For example, if we consider
the following JSON object:

{

    "price":12.12 

}

the following test will fail, because we compare with a "double" instead of a "float":

get("/price").then().assertThat().body("price", equalTo(12.12));

Instead, compare with a float with:

get("/price").then().assertThat().body("price", equalTo(12.12f));

## Note on syntax

When reading blogs about REST Assured you may see a lot of examples using the "given / expect /
when" syntax, for example:

given().
        param("x", "y").
expect().
        body("lotto.lottoId", equalTo(5)).
when().
        get("/lotto");

This is the so called "legacy syntax" which was the de facto way of writing tests in REST Assured
1.x. While this works fine it turned out to be quite confusing and annoying for many users. The
reason for not using "given / when / then" in the first place was mainly technical. So prior to REST
Assured 2.0 there was no support "given / when / then" which is more or less the standard approach
when you're doing some kind of BDD-like testing. The "given / expect / when" approach still works
fine in 2.0 and above but "given / when / then" reads better and is easier to understand for most
people and is thus recommended in most cases. There's however one benefit of using the "given /
expect / when" approach and that is that ALL expectation errors can be displayed at the same time
which is not possible with the new syntax (since the expectations are defined last). This means that
if you would have had multiple expectations in the previous example such as

given().
        param("x", "y").
expect().
        statusCode(400).
        body("lotto.lottoId", equalTo(6)).
when().
        get("/lotto");

REST Assured will report that both the status code expectation and the body expectation are wrong.
Rewriting this with the new syntax

given().
        param("x", "y").
when().
        get("/lotto").
then().
        statusCode(400).
        body("lotto.lottoId", equalTo(6));

will only report an error at the first failed expectation / assertion (that status code was expected
to be 400 but it was actually 200). You would have to re-run the test in order to catch the second
error.

### Syntactic Sugar

Another thing worth mentioning is that REST Assured contains some methods that are only there for
syntactic sugar. For example the "and" method which can add readability if you're writing everything
in a one-liner, for example:

given().param("x", "y").and().header("z", "w").when().get("/something").then().assertThat().statusCo
de(200).and().body("x.y", equalTo("z"));

This is the same thing as:

given().
        param("x", "y").
        header("z", "w").
when().
        get("/something").
then().
        statusCode(200).
        body("x.y", equalTo("z"));

# Getting Response Data

You can also get the content of a response. E.g. let's say you want to return the body of a get
request to "/lotto". You can get it a variety of different ways:

InputStream stream = get("/lotto").asInputStream(); // Don't forget to close this one when you're do
ne
byte[] byteArray = get("/lotto").asByteArray();
String json = get("/lotto").asString();

## Extracting values from the Response after validation

You can extract values from the response or return the response instance itself after you've done
validating the response by using the `extract` method. This is useful for example if you want to use
values from the response in sequent requests. For example given that a resource called `title`
returns the following JSON

 {
     "title" : "My Title",
      "_links": {
              "self": { "href": "/title" },
              "next": { "href": "/title?page=2" }
           }
 }

and you want to validate that content type is equal to `JSON` and the title is equal to `My Title`
but you also want to extract the link to the `next` title to use that in a subsequent request. This
is how:

String nextTitleLink =
given().
        param("param_name", "param_value").
when().
        get("/title").
then().
        contentType(JSON).
        body("title", equalTo("My Title")).
extract().
        path("_links.next.href");

get(nextTitleLink). ..

You could also decide to instead return the entire response if you need to extract multiple values
from the response:

Response response = 
given().
        param("param_name", "param_value").
when().
        get("/title").
then().
        contentType(JSON).
        body("title", equalTo("My Title")).
extract().
        response(); 

String nextTitleLink = response.path("_links.next.href");
String headerValue = response.header("headerName");

## JSON (using JsonPath)

Once we have the response body we can then use the [JsonPath][191] to get data from the response
body:

int lottoId = from(json).getInt("lotto.lottoId");
List<Integer> winnerIds = from(json).get("lotto.winners.winnerId");

Or a bit more efficiently:

JsonPath jsonPath = new JsonPath(json).setRoot("lotto");
int lottoId = jsonPath.getInt("lottoId");
List<Integer> winnerIds = jsonPath.get("winners.winnderId");

Note that you can use `JsonPath` standalone without depending on REST Assured, see [getting started
guide][192] for more info on this.

### JsonPath Configuration

You can configure object de-serializers etc for JsonPath by configuring it, for example:

JsonPath jsonPath = new JsonPath(SOME_JSON).using(new JsonPathConfig("UTF-8"));

It's also possible to configure JsonPath statically so that all instances of JsonPath will shared
the same configuration:

JsonPath.config = new JsonPathConfig("UTF-8");

You can read more about JsonPath at [this blog][193].

Note that the JsonPath implementation uses [Groovy's GPath][194] syntax and is not to be confused
with Jayway's [JsonPath][195] implementation.

## XML (using XmlPath)

You also have the corresponding functionality for XML using [XmlPath][196]:

String xml = post("/greetXML?firstName=John&lastName=Doe").andReturn().asString();
// Now use XmlPath to get the first and last name
String firstName = from(xml).get("greeting.firstName");
String lastName = from(xml).get("greeting.firstName");

// or a bit more efficiently:
XmlPath xmlPath = new XmlPath(xml).setRoot("greeting");
String firstName = xmlPath.get("firstName");
String lastName = xmlPath.get("lastName");

Note that you can use `XmlPath` standalone without depending on REST Assured, see [getting started
guide][197] for more info on this.

### XmlPath Configuration

You can configure object de-serializers and charset for XmlPath by configuring it, for example:

XmlPath xmlPath = new XmlPath(SOME_XML).using(new XmlPathConfig("UTF-8"));

It's also possible to configure XmlPath statically so that all instances of XmlPath will shared the
same configuration:

XmlPath.config = new XmlPathConfig("UTF-8");

You can read more about XmlPath at [this blog][198].

### Parsing HTML with XmlPath

By configuring XmlPath with [compatibility mode][199] `HTML` you can also use the XmlPath syntax
(Gpath) to parse HTML pages. For example if you want to extract the title of this HTML document:

<html>
<head>
    <title>my title</title>
  </head>
  <body>
    <p>paragraph 1</p>
     <br>
    <p>paragraph 2</p>
  </body>
</html>

you can configure XmlPath like this:

String html = ...
XmlPath xmlPath = new XmlPath(CompatibilityMode.HTML, html);

and then extract the title like this:

xmlPath.getString("html.head.title"); // will return "mytitle"

In this example we've statically imported: `io.restassured.path.xml.XmlPath.CompatibilityMode.HTML`;

## Single path

If you only want to make a request and return a single path you can use a shortcut:

int lottoId = get("/lotto").path("lotto.lottoid");

REST Assured will automatically determine whether to use JsonPath or XmlPath based on the
content-type of the response. If no content-type is defined then REST Assured will try to look at
the [default parser][200] if defined. You can also manually decide which path instance to use, e.g.

String firstName = post("/greetXML?firstName=John&lastName=Doe").andReturn().xmlPath().getString("fi
rstName");

Options are `xmlPath`, `jsonPath` and `htmlPath`.

## Headers, cookies, status etc

You can also get headers, cookies, status line and status code:

Response response = get("/lotto");

// Get all headers
Headers allHeaders = response.getHeaders();
// Get a single header value:
String headerName = response.getHeader("headerName");

// Get all cookies as simple name-value pairs
Map<String, String> allCookies = response.getCookies();
// Get a single cookie value:
String cookieValue = response.getCookie("cookieName");

// Get status line
String statusLine = response.getStatusLine();
// Get status code
int statusCode = response.getStatusCode();

## Multi-value headers and cookies

A header and a cookie can contain several values for the same name.

### Multi-value headers

To get all values for a header you need to first get the [Headers][201] object from the
[Response][202] object. From the `Headers` instance you can get all values using the
[Headers.getValues(][203]

) method which returns a `List` with all header values.

### Multi-value cookies

To get all values for a cookie you need to first get the [Cookies][204] object from the
[Response][205] object. From the `Cookies` instance you can get all values using the
[Cookies.getValues()][206] method which returns a `List` with all cookie values.

## Detailed Cookies

If you need to get e.g. the comment, path or expiry date etc from a cookie you need get a [detailed
cookie][207] from REST Assured. To do this you can use the
[Response.getDetailedCookie(java.lang.String)][208] method. The detailed cookie then contains all
attributes from the cookie.

You can also get all detailed response [cookies][209] using the [Response.getDetailedCookies()][210]
method.

# Specifying Request Data

Besides specifying request parameters you can also specify headers, cookies, body and content type.

## Invoking HTTP resources

You typically perform a request by calling any of the "HTTP methods" in the [request
specification][211]. For example:

when().get("/x"). ..;

Where `get` is the HTTP request method.

As of REST Assured 3.0.0 you can use any HTTP verb with your request by making use of the
[request][212] method.

when().
       request("CONNECT", "/somewhere").
then().
       statusCode(200);

This will send a "connect" request to the server.

## Parameters

Normally you specify parameters like this:

given().
       param("param1", "value1").
       param("param2", "value2").
when().
       get("/something");

REST Assured will automatically try to determine which parameter type (i.e. query or form parameter)
based on the HTTP method. In case of GET query parameters will automatically be used and in case of
POST form parameters will be used. In some cases it's however important to separate between form and
query parameters in a PUT or POST. You can then do like this:

given().
       formParam("formParamName", "value1").
       queryParam("queryParamName", "value2").
when().
       post("/something");

Parameters can also be set directly on the url:

..when().get("/name?firstName=John&lastName=Doe");

For multi-part parameters please refer to the [Multi-part form data][213] section.

### Multi-value parameter

Multi-value parameters are parameters with more then one value per parameter name (i.e. a list of
values per name). You can specify these either by using var-args:

given().param("myList", "value1", "value2"). .. 

or using a list:

List<String> values = new ArrayList<String>();
values.add("value1");
values.add("value2");

given().param("myList", values). .. 

### No-value parameter

You can also specify a query, request or form parameter without a value at all:

given().param("paramName"). ..

### Path parameters

You can also specify so called path parameters in your request, e.g.

post("/reserve/{hotelId}/{roomNumber}", "My Hotel", 23);

These kinds of path parameters are referred to "unnamed path parameters" in REST Assured since they
are index based (`hotelId` will be equal to "My Hotel" since it's the first placeholder).

You can also use named path parameters:

given().
        pathParam("hotelId", "My Hotel").
        pathParam("roomNumber", 23).
when(). 
        post("/reserve/{hotelId}/{roomNumber}").
then().
         ..

Path parameters makes it easier to read the request path as well as enabling the request path to
easily be re-usable in many tests with different parameter values.

As of version 2.8.0 you can mix unnamed and named path parameters:

given().
        pathParam("hotelId", "My Hotel").        
when(). 
        post("/reserve/{hotelId}/{roomNumber}", 23).
then().
         ..

Here `roomNumber` will be replaced with `23`.

Note that specifying too few or too many parameters will result in an error message. For advanced
use cases you can add, change, remove (even redundant path parameters) from a [filter][214].

## Cookies

In its simplest form you specify cookies like this:

given().cookie("username", "John").when().get("/cookie").then().body(equalTo("username"));

You can also specify a multi-value cookie like this:

given().cookie("cookieName", "value1", "value2"). ..

This will create *two* cookies, `cookieName=value1` and `cookieName=value2`.

You can also specify a detailed cookie using:

Cookie someCookie = new Cookie.Builder("some_cookie", "some_value").setSecured(true).setComment("som
e comment").build();
given().cookie(someCookie).when().get("/cookie").then().assertThat().body(equalTo("x"));

or several detailed cookies at the same time:

Cookie cookie1 = Cookie.Builder("username", "John").setComment("comment 1").build();
Cookie cookie2 = Cookie.Builder("token", 1234).setComment("comment 2").build();
Cookies cookies = new Cookies(cookie1, cookie2);
given().cookies(cookies).when().get("/cookie").then().body(equalTo("username, token"));

## Headers

given().header("MyHeader", "Something").and(). ..
given().headers("MyHeader", "Something", "MyOtherHeader", "SomethingElse").and(). ..

You can also specify a multi-value headers like this:

given().header("headerName", "value1", "value2"). ..

This will create *two* headers, `headerName: value1` and `headerName: value2`.

#### Header Merging/Overwriting

By default headers are merged. So for example if you do like this:

given().header("x", "1").header("x", "2"). ..

The request will contain two headers, "x: 1" and "x: 2". You can change in this on a per header
basis in the [HeaderConfig][215]. For example:

given().
        config(RestAssuredConfig.config().headerConfig(headerConfig().overwriteHeadersWithName("x"))
).
        header("x", "1").
        header("x", "2").
when().
        get("/something").
...

This means that only one header, "x: 2", is sent to server.

## Content Type

given().contentType(ContentType.TEXT). ..
given().contentType("application/json"). ..

You can also instruct Rest Assured to don't include any (default) content-type at all:

given().noContentType(). ..

## Request Body

given().body("some body"). .. // Works for POST, PUT and DELETE requests
given().request().body("some body"). .. // More explicit (optional)
given().body(new byte[]{42}). .. // Works for POST, PUT and DELETE
given().request().body(new byte[]{42}). .. // More explicit (optional)

You can also serialize a Java object to JSON or XML. Click [here][216] for details.

# Verifying Response Data

You can also verify status code, status line, cookies, headers, content type and body.

## Response Body

See Usage examples, e.g. [JSON][217] or [XML][218].

You can also map a response body to a Java Object, click [here][219] for details.

## Cookies

get("/x").then().assertThat().cookie("cookieName", "cookieValue"). ..
get("/x").then().assertThat().cookies("cookieName1", "cookieValue1", "cookieName2", "cookieValue2").
 ..
get("/x").then().assertThat().cookies("cookieName1", "cookieValue1", "cookieName2", containsString("
Value2")). ..

## Status

get("/x").then().assertThat().statusCode(200). ..
get("/x").then().assertThat().statusLine("something"). ..
get("/x").then().assertThat().statusLine(containsString("some")). ..

## Headers

get("/x").then().assertThat().header("headerName", "headerValue"). ..
get("/x").then().assertThat().headers("headerName1", "headerValue1", "headerName2", "headerValue2").
 ..
get("/x").then().assertThat().headers("headerName1", "headerValue1", "headerName2", containsString("
Value2")). ..

It's also possible to use a mapping function when validating headers. For example let's say you want
to validate that the `Content-Length` header is less than 1000. You can then use a mapping function
to first convert the header value to an int and then use an `Integer` before validating it with a
Hamcrest matcher:

get("/something").then().assertThat().header("Content-Length", Integer::parseInt, lessThan(1000));

## Content-Type

get("/x").then().assertThat().contentType(ContentType.JSON). ..

## Full body/content matching

get("/x").then().assertThat().body(equalTo("something")). ..

## Use the response to verify other parts of the response

You can use data from the response to verify another part of the response. For example consider the
following JSON document returned from service x:

{ "userId" : "some-id", "href" : "http://localhost:8080/some-id" }

You may notice that the "href" attribute ends with the value of the "userId" attribute. If we want
to verify this we can implement a `io.restassured.matcher.ResponseAwareMatcher` and use it like
this:

get("/x").then().body("href", new ResponseAwareMatcher<Response>() {
                                  public Matcher<?> matcher(Response response) {
                                          return equalTo("http://localhost:8080/" + response.path("u
serId"));
                                  }
                       });

If you're using Java 8 you can use a lambda expression instead:

get("/x").then().body("href", response -> equalTo("http://localhost:8080/" + response.path("userId")
);

There are some predefined matchers that you can use defined in the
`io.restassured.matcher.RestAssuredMatchers` (or
`io.restassured.module.mockmvc.matcher.RestAssuredMockMvcMatchers` if using the spring-mock-mvc
module). For example:

get("/x").then().body("href", endsWithPath("userId"));

`ResponseAwareMatchers` can also be composed, either with another `ResponseAwareMatcher` or with a
Hamcrest Matcher. For example:

get("/x").then().body("href", and(startsWith("http:/localhost:8080/"), endsWithPath("userId")));

The `and` method is statically imported from `io.restassured.matcher.ResponseAwareMatcherComposer`.

## Measuring Response Time

As of version 2.8.0 REST Assured has support measuring response time. For example:

long timeInMs = get("/lotto").time()

or using a specific time unit:

long timeInSeconds = get("/lotto").timeIn(SECONDS);

where `SECONDS` is just a standard `TimeUnit`. You can also validate it using the validation DSL:

when().
      get("/lotto").
then().
      time(lessThan(2000L)); // Milliseconds

or

when().
      get("/lotto").
then().
      time(lessThan(2L), SECONDS);

Please note that response time measurement should be performed when the JVM is hot! (i.e. running a
response time measurement when only running a single test will yield erroneous results). Also note
that you can only vaguely regard these measurments to correlate with the server request processing
time (since the response time will include the HTTP round trip and REST Assured processing time
among other things).

# Authentication

REST assured also supports several authentication schemes, for example OAuth, digest, certificate,
form and preemptive basic authentication. You can either set authentication for each request:

given().auth().basic("username", "password"). ..

but you can also define authentication for all requests:

RestAssured.authentication = basic("username", "password");

or you can use a [specification][220].

## Basic Authentication

There are two types of basic authentication, preemptive and "challenged basic authentication".

### Preemptive Basic Authentication

This will send the basic authentication credential even before the server gives an unauthorized
response in certain situations, thus reducing the overhead of making an additional connection. This
is typically what you want to use in most situations unless you're testing the servers ability to
challenge. Example:

given().auth().preemptive().basic("username", "password").when().get("/secured/hello").then().status
Code(200);

### Challenged Basic Authentication

When using "challenged basic authentication" REST Assured will not supply the credentials unless the
server has explicitly asked for it. This means that REST Assured will make an additional request to
the server in order to be challenged and then follow up with the same request once more but this
time setting the basic credentials in the header.

given().auth().basic("username", "password").when().get("/secured/hello").then().statusCode(200);

## Digest Authentication

Currently only "challenged digest authentication" is supported. Example:

given().auth().digest("username", "password").when().get("/secured"). ..

## Form Authentication

[Form authentication][221] is very popular on the internet. It's typically associated with a user
filling out his credentials (username and password) on a webpage and then pressing a login button of
some sort. A very simple HTML page that provide the basis for form authentication may look like
this:

<html>
  <head>
    <title>Login</title>
  </head>

  <body>
    <form action="j_spring_security_check" method="POST">
      <table>
        <tr><td>User:&nbsp;</td><td><input type='text' name='j_username'></td></tr>
        <tr><td>Password:</td><td><input type='password' name='j_password'></td></tr>
          <tr><td colspan='2'><input name="submit" type="submit"/></td></tr>
       </table>
        </form>
      </body>
 </html>

I.e. the server expects the user to fill-out the "j_username" and "j_password" input fields and then
press "submit" to login. With REST Assured you can test a service protected by form authentication
like this:

given().
        auth().form("John", "Doe").
when().
        get("/formAuth");
then().
        statusCode(200);

While this may work it's not optimal. What happens when form authentication is used like this in
REST Assured an additional request have to made to the server in order to retrieve the webpage with
the login details. REST Assured will then try to parse this page and look for two input fields (with
username and password) as well as the form action URI. This may work or fail depending on the
complexity of the webpage. A better option is to supply the these details when setting up the form
authentication. In this case one could do:

given().
        auth().form("John", "Doe", new FormAuthConfig("/j_spring_security_check", "j_username", "j_p
assword")).
when().
        get("/formAuth");
then().
        statusCode(200);

This way REST Assured doesn't need to make an additional request and parse the webpage. There's also
a predefined FormAuthConfig called `springSecurity` that you can use if you're using the default
Spring Security properties:

given().
        auth().form("John", "Doe", FormAuthConfig.springSecurity()).
when().
        get("/formAuth");
then().
        statusCode(200);

### CSRF

Today it's common for the server to supply a [CSRF][222] token with the response in order to avoid
these kinds of attacks. REST Assured has support for automatically parsing and supplying the CSRF
token to the server. In order for this to work REST Assured *must* make an additional request and
parse (parts) of the website.

REST Assured supports two ways of providing CSRF tokens to the server, either by submitting the CSRF
token in a form, or as a header.

#### CSRF Form Token

Consider that we have a simple website that adds a user:

<html>
<head>
<title>Add User</title>
</head>
<body>
<form action="/users" method="POST">
<table>
  <tr>
      <td>First Name:&nbsp;</td>
      <td><input type="text" name="firstName"></td>
  </tr>
  <tr>
      <td>Last Name:</td>
      <td><input type="text" name="lastName"></td>
  </tr>
  <tr>
      <td colspan="2"><input name="submit" type="submit"/></td>
  </tr>
</table>
<input type="hidden" name="_csrf" value="8adf2ea1-b246-40aa-8e13-a85fb7914341"/>
</form>
</body>
</html>

We can see that the form contains a `hidden` input field with a CSRF token. If we want to test
posting a new user to `/users` we need to include this CSRF token in the request. We can do this by
doing:

given().
        csrf("/users", "_csrf")
        formParam("firstName", "John")
        formParam("lastName", "Doe")
when().
        post("/users").
then().
        statusCode(200);

REST Assured will now first perform a `GET` request to `/users` (the first argument to the `csrf`
method) and it expects the form to have a (hidden) input field named `_csrf` (second argument to the
`csrf` method). REST Assured will then find `8adf2ea1-b246-40aa-8e13-a85fb7914341` and include it as
a form paramter in the `POST` request to `/users`.

Since `_csrf` is the default expect input field name for CSRF tokens in REST Assured, we could have
skipped it and just do:

given().
        csrf("/users")
        formParam("firstName", "John")
        formParam("lastName", "Doe")
when().
        post("/users").
then().
        statusCode(200);

which would work as well. You can configure the default CSRF form token name in the config, e.g. to
change the default name to `_mycsrf` you can do: `RestAssured.config =
RestAssuredConfig.config().csrfConfig(csrfConfig().with().csrfInputFieldName("_mycsrf"));`

The config also supports, among other things, to set a default resource to get CSRF token from for
all requests.

Note that it's common to get the CSRF token from one resource but `POST` to another resource. Let's
say that, in the example above, the "add user" form is located at `/users/form`, you can do:

given().
        csrf("/users/form")
        formParam("firstName", "John")
        formParam("lastName", "Doe")
when().
        post("/users").
then().
        statusCode(200);

#### CSRF Header Token

Besides sending CSRF tokens in forms, REST Assured also support sending a CSRF token in a header.
For example, if you have a login page at `/login` that looks like this:

<html>
<head>
  <title>Login</title>
  <meta name="_csrf_header" content="ab8722b1-1f23-4dcf-bf63-fb8b94be4107"/>
</head>
<body>
   ..
</body>
</html>

The csrf meta tag name is called `_csrf_header` (which is the default meta tag name used by REST
Assured). You can do like this to POST to `/pageThatRequireHeaderCsrf` and make REST Assured include
the CSRF token (`ab8722b1-1f23-4dcf-bf63-fb8b94be4107`) in a header called `X-CSRF-TOKEN`:

given().
        csrf("/login").
when().
        post("/pageThatRequireHeaderCsrf").
then().
        statusCode(200);

You can configure a different the meta name (if it's not called `_csrf_header`) like this:

given().
        config(RestAssuredConfig.config().csrfConfig(csrfConfig().csrfMetaTagName("_my_csrf_header")
)).
        csrf("/login").
when().
        post("/pageThatRequireHeaderCsrf").
then().
        statusCode(200);

Additionally, you can change the header name:

`given().
        config(RestAssuredConfig.config().csrfConfig(csrfConfig().csrfHeaderName("MyHeader"))).
        csrf("/login").
when().
        post("/pageThatRequireHeaderCsrf").
then().
        statusCode(200);
`

REST Assured will now send the CSRF token in a header called `MyHeader` instead of the default,
`X-CSRF-TOKEN`.

#### CSRF and Authentication

The page you GET to extract the CSRF token might be protected by authentication. REST Assured
automatically applies authentication to the CSRF resource as well if defined in the DSL. For
example, let's say that the `/users` (see above) resources requires basic authentication for both
GET and POST. You can then specify authentication as you normally would and this would be applied to
the CSRF request as well:

given().
        auth().preemptive().basic("username", "password")
        csrf("/users")
        formParam("firstName", "John")
        formParam("lastName", "Doe")
when().
        post("/users").
then().
        statusCode(200);

#### CSRF Prioritization

If the page you use to get the CSRF token both contains a header token *and* a (different) CSRF form
token you can instruct REST Assured which one to prioritize:

given().config(RestAssuredConfig.config().csrfConfig(csrfConfig().csrfPrioritization(CsrfPrioritizat
ion.HEADER))). .. 

Change `CsrfPrioritization.HEADER` to `CsrfPrioritization.FORM` to prioritize form tokens instead.

#### CSRF Cookie Propagation

As of version 5.5.3, REST Assured automatically forwards the cookies from the GET request that
returns the CSRF token and applies them to the "actual" request. These cookies will also be applied
to the `CookieFilter` automatically (if configured) and [SessionFilter][223] (if configured). For
example:

given().
    csrf("/login").
    formParam("name", "My New Name").
when().
    post("/users/123").
then().
    statusCode(200);

Now the cookies returned from the GET request to login will be automatically applied to the POST to
"/users/123".

If you have a CookieFilter defined for multiple requests, the cookies returned by GET to /login will
be automatically stored in the CookieFilter and used in the second request.

var cookieFilter = new CookieFilter()
given().
        filter(cookieFilter).
        csrf("/login").
        formParam("name", "My New Name").
when().
        post("/users/123").
then().
        statusCode(200);

given().
        filter(cookieFilter).
when().
        get("/users/123").
then().
        statusCode(200);

You can disable this behavior by setting automaticallyApplyCookies to false the csrf config:

`given().
        config(config().csrfConfig(csrfConfig().automaticallyApplyCookies(false))).
        csrf("/login").
when().
        ...
`

## OAuth

In order to use OAuth 1 and OAuth 2 (for query parameter signing) you need to add [Scribe][224] to
your classpath (if you're using version 2.1.0 or older of REST Assured then please refer to the
[legacy][225] documentation). In Maven you can simply add the following dependency:

<dependency>
            <groupId>com.github.scribejava</groupId>
            <artifactId>scribejava-apis</artifactId>
            <version>2.5.3</version>
            <scope>test</scope>
</dependency>

If you're not using Maven [download][226] a Scribe release manually and put it in your classpath.

### OAuth 1

OAuth 1 requires [Scribe][227] in the classpath. To use auth 1 authentication you can do:

given().auth().oauth(..). ..

### OAuth 2

Since version `2.5.0` you can use OAuth 2 authentication without depending on [Scribe][228]:

given().auth().oauth2(accessToken). ..

This will put the OAuth2 `accessToken` in a header. To be more explicit you can also do:

given().auth().preemptive().oauth2(accessToken). ..

There reason why `given().auth().oauth2(..)` still exists is for backward compatibility (they do the
same thing). If you need to provide the OAuth2 token in a query parameter you currently need
[Scribe][229] in the classpath. Then you can do like this:

given().auth().oauth2(accessToken, OAuthSignature.QUERY_STRING). ..

## Custom Authentication

Rest Assured allows you to create custom authentication providers. You do this by implementing the
`io.restassured.spi.AuthFilter` interface (preferably) and apply it as a [filter][230]. For example
let's say that your security consists of adding together two headers together in a new header called
"AUTH" (this is of course not secure). Then you can do that like this (Java 8 syntax):

given().
        filter((requestSpec, responseSpec, ctx) -> {
            String header1 = requestSpec.getHeaders().getValue("header1");
            String header2 = requestSpec.getHeaders().getValue("header2");
            requestSpec.header("AUTH", header1 + header2);
            return ctx.next(requestSpec, responseSpec);
        }).
when().
        get("/customAuth").
then().
  statusCode(200);

The reason why you want to use a `AuthFilter` and not `Filter` is that `AuthFilters` are
automatically removed when doing `given().auth().none(). ..`.

# Multi-part form data

When sending larger amount of data to the server it's common to use the multipart form data
technique. Rest Assured provide methods called `multiPart` that allows you to specify a file,
byte-array, input stream or text to upload. In its simplest form you can upload a file like this:

given().
        multiPart(new File("/path/to/file")).
when().
        post("/upload");

It will assume a control name called "file". In HTML the control name is the attribute name of the
input tag. To clarify let's look at the following HTML form:

<form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" size="40">
        <input type=submit value="Upload!">
</form>

The control name in this case is the name of the input tag with name "file". If you have a different
control name then you need to specify it:

given().
        multiPart("controlName", new File("/path/to/file")).
when().
        post("/upload");

It's also possible to supply multiple "multi-parts" entities in the same request:

byte[] someData = ..
given().
        multiPart("controlName1", new File("/path/to/file")).
        multiPart("controlName2", "my_file_name.txt", someData).
        multiPart("controlName3", someJavaObject, "application/json").
when().
        post("/upload");

For more advanced use cases you can make use of the [MultiPartSpecBuilder][231]. For example:

Greeting greeting = new Greeting();
greeting.setFirstName("John");
greeting.setLastName("Doe");

given().
        multiPart(new MultiPartSpecBuilder(greeting, ObjectMapperType.JACKSON_2)
                .fileName("greeting.json")
                .controlName("text")
                .mimeType("application/vnd.custom+json").build()).
when().
        post("/multipart/json").
then().
        statusCode(200);

You can specify, among other things, the default `control name` and filename using the
[MultiPartConfig][232]. For example:

given().config(config().multiPartConfig(multiPartConfig().defaultControlName("something-else"))). ..

This will configure the default control name to be "something-else" instead of "file".

For additional info refer to [this][233] blog post.

# Object Mapping

REST Assured supports mapping Java objects to and from JSON and XML. For JSON you need to have
either Jackson, Jackson2, Jackson3, Gson, Yasson or Johnzon in the classpath and for XML you need
Jakarta EE or JAXB.

## Jakarta EE

To use Jakarta EE for XML object mapping you need to include the following dependencies:

<dependency>
    <groupId>jakarta.xml.bind</groupId>
    <artifactId>jakarta.xml.bind-api</artifactId>
    <version>3.0.1</version>
</dependency>
<dependency>
    <groupId>org.glassfish.jaxb</groupId>
    <artifactId>jaxb-runtime</artifactId>
    <version>3.0.2</version>
</dependency>

## JAXB

JAXB can be used to serialize/deserialize XML with REST Assured. From version 4.5.0 it is not
included by default, so if you're using a newer version of Java, such as Java 17, you need to add
these two dependencies or use [Jakarta EE][234]:

<dependency>
    <groupId>javax.xml.bind</groupId>
    <artifactId>jaxb-api</artifactId>
    <version>2.3.1</version>
</dependency>
<dependency>
    <groupId>com.sun.xml.bind</groupId>
    <artifactId>jaxb-impl</artifactId>
    <version>2.3.4</version>
</dependency>

## Serialization

Let's say we have the following Java object:

public class Message {
    private String message;

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }
}

and you want to serialize this object to JSON and send it with the request. There are several ways
to do this, e.g:

### Content-Type based Serialization

Message message = new Message();
message.setMessage("My messagee");
given().
       contentType("application/json").
       body(message).
when().
      post("/message");

In this example REST Assured will serialize the object to JSON since the request content-type is set
to "application/json". It will first try to use Jackson if found in classpath and if not Gson will
be used. If you change the content-type to "application/xml" REST Assured will serialize to XML
using JAXB. If no content-type is defined REST Assured will try to serialize in the following order:

1. JSON using Jackson 3 (databind)
2. JSON using Jackson 2 (Faster Jackson (databind))
3. JSON using Jackson (databind)
4. JSON using Gson
5. JSON using Johnzon
6. JSON-B using Eclipse Yasson
7. XML using Jakarta EE
8. XML using JAXB

REST Assured also respects the charset of the content-type. E.g.

Message message = new Message();
message.setMessage("My messagee");
given().
       contentType("application/json; charset=UTF-16").
       body(message).
when().
      post("/message");

You can also serialize the `Message` instance as a form parameter:

Message message = new Message();
message.setMessage("My messagee");
given().
       contentType("application/json; charset=UTF-16").
       formParam("param1", message).
when().
      post("/message");

The message object will be serialized to JSON using Jackson (databind) (if present) or Gson (if
present) with UTF-16 encoding.

### Create JSON from a HashMap

You can also create a JSON document by supplying a Map to REST Assured.

Map<String, Object>  jsonAsMap = new HashMap<>();
jsonAsMap.put("firstName", "John");
jsonAsMap.put("lastName", "Doe");

given().
        contentType(JSON).
        body(jsonAsMap).
when().
        post("/somewhere").
then().
        statusCode(200);

This will provide a JSON payload as:

{ "firstName" : "John", "lastName" : "Doe" }

### Using an Explicit Serializer

If you have multiple object mappers in the classpath at the same time or don't care about setting
the content-type you can specify a serializer explicity. E.g.

Message message = new Message();
message.setMessage("My messagee");
given().
       body(message, ObjectMapperType.JAXB).
when().
      post("/message");

In this example the Message object will be serialized to XML using JAXB.

## Deserialization

Again let's say we have the following Java object:

public class Message {
    private String message;

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }
}

and we want the response body to be deserialized into a Message object.

### Content-Type based Deserialization

Let's assume then that the server returns a JSON body like this:

{"message":"My message"}

To deserialize this to a Message object we simply to like this:

Message message = get("/message").as(Message.class);

For this to work the response content-type must be "application/json" (or something that contains
"json"). If the server instead returned

<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<message>
      <message>My message</message>
</message>

and a content-type of "application/xml" you wouldn't have to change the code at all:

Message message = get("/message").as(Message.class);

#### Custom Content-Type Deserialization

If the server returns a custom content-type, let's say "application/something", and you still want
to use the object mapping in REST Assured there are a couple of different ways to go about. You can
either use the [explicit][235] approach or register a parser for the custom content-type:

Message message = expect().parser("application/something", Parser.XML).when().get("/message").as(Mes
sage.class);

or

Message message = expect().defaultParser(Parser.XML).when().get("/message").as(Message.class);

You can also register a default or custom parser [statically][236] or using [specifications][237].

### Using an Explicit Deserializer

If you have multiple object mappers in the classpath at the same time or don't care about the
response content-type you can specify a deserializer explicitly. E.g.

Message message = get("/message").as(Message.class, ObjectMapperType.GSON);

## Configuration

You can configure the pre-defined object mappers by using a [ObjectMapperConfig][238] and pass it to
[detailed configuration][239]. For example to change GSON to use lower case with underscores as
field naming policy you can do like this:

RestAssured.config = RestAssuredConfig.config().objectMapperConfig(objectMapperConfig().gsonObjectMa
pperFactory(
                new GsonObjectMapperFactory() {
                    public Gson create(Class cls, String charset) {
                        return new GsonBuilder().setFieldNamingPolicy(LOWER_CASE_WITH_UNDERSCORES).c
reate();
                    }
                }
        ));

There are pre-defined object mapper factories for GSON, JAXB, Jackson (1-3), and Eclipse Yasson
(JSON-B).

## Custom

By default REST Assured will scan the classpath to find various object mappers. If you want to
integrate an object mapper that is not supported by default or if you've rolled your own you can
implement the [io.restassured.mapper.ObjectMapper][240] interface. You tell REST Assured to use your
object mapper either by passing it as a second parameter to the body:

given().body(myJavaObject, myObjectMapper).when().post("..")

or you can define it statically once and for all:

RestAssured.config = RestAssuredConfig.config().objectMapperConfig(new ObjectMapperConfig(myObjectMa
pper));

For an example see [here][241].

# Custom parsers

REST Assured providers predefined parsers for e.g. HTML, XML and JSON. But you can parse other kinds
of content by registering a predefined parser for unsupported content-types by using:

RestAssured.registerParser(<content-type>, <parser>);

E.g. to register that mime-type 'application/vnd.uoml+xml' should be parsed using the XML parser do:

RestAssured.registerParser("application/vnd.uoml+xml", Parser.XML);

You can also unregister a parser using:

RestAssured.unregisterParser("application/vnd.uoml+xml");

Parsers can also be specified per "request":

get(..).then().using().parser("application/vnd.uoml+xml", Parser.XML). ..;

and using a [response specification][242].

# Default parser

Sometimes it's useful to specify a default parser, e.g. if the response doesn't contain a
content-type at all:

RestAssured.defaultParser = Parser.JSON;

You can also specify a default parser for a single request:

get("/x").then().using().defaultParser(Parser.JSON). ..

or using a [response specification][243].

# Default values

By default REST assured assumes host localhost and port 8080 when doing a request. If you want a
different port you can do:

given().port(80). ..

or simply:

..when().get("http://myhost.org:80/doSomething");

You can also change the default base URI, base path, port and authentication scheme for all
subsequent requests:

RestAssured.baseURI = "http://myhost.org";
RestAssured.port = 80;
RestAssured.basePath = "/resource";
RestAssured.authentication = basic("username", "password");
RestAssured.rootPath = "x.y.z";

This means that a request like e.g. `get("/hello")` goes to:
[http://myhost.org:80/resource/hello][244] with basic authentication credentials "username" and
"password". See [rootPath][245] for more info about setting the root paths. Other default values you
can specify are:

RestAssured.filters(..); // List of default filters
RestAssured.requestSpecification = .. // Default request specification
RestAssured.responseSpecification = .. // Default response specification
RestAssured.urlEncodingEnabled = .. // Specify if Rest Assured should URL encoding the parameters
RestAssured.defaultParser = .. // Specify a default parser for response bodies if no registered pars
er can handle data of the response content-type
RestAssured.registerParser(..) // Specify a parser for the given content-type
RestAssured.unregisterParser(..) // Unregister a parser for the given content-type

You can reset to the standard baseURI (localhost), basePath (empty), standard port (8080), standard
root path (""), default authentication scheme (none) and url encoding enabled (true) using:

RestAssured.reset();

# Specification Re-use

Instead of having to duplicate response expectations and/or request parameters for different tests
you can re-use an entire specification. To do this you define a specification using either the
[RequestSpecBuilder][246] or [ResponseSpecBuilder][247].

E.g. let's say you want to make sure that the expected status code is 200 and that the size of the
JSON array "x.y" has size 2 in several tests you can define a ResponseSpecBuilder like this:

ResponseSpecBuilder builder = new ResponseSpecBuilder();
builder.expectStatusCode(200);
builder.expectBody("x.y.size()", is(2));
ResponseSpecification responseSpec = builder.build();

// Now you can re-use the "responseSpec" in many different tests:
when().
       get("/something").
then().
       spec(responseSpec).
       body("x.y.z", equalTo("something"));

In this example the data defined in "responseSpec" is merged with the additional body expectation
and all expectations must be fulfilled in order for the test to pass.

You can do the same thing if you need to re-use request data in different tests. E.g.

RequestSpecBuilder builder = new RequestSpecBuilder();
builder.addParam("parameter1", "parameterValue");
builder.addHeader("header1", "headerValue");
RequestSpecification requestSpec = builder.build();
  
given().
        spec(requestSpec).
        param("parameter2", "paramValue").
when().
        get("/something").
then().
        body("x.y.z", equalTo("something"));        

Here the request's data is merged with the data in the "requestSpec" so the request will contain two
parameters ("parameter1" and "parameter2") and one header ("header1").

## Querying RequestSpecification

Sometimes it's useful to be able to query/extract values form a RequestSpecification. For this
reason you can use the `io.restassured.specification.SpecificationQuerier`. For example:

RequestSpecification spec = ...
QueryableRequestSpecification queryable = SpecificationQuerier.query(spec);
String headerValue = queryable.getHeaders().getValue("header");
String param = queryable.getFormParams().get("someparam");

# Filters

A filter allows you to inspect and alter a request before it's actually committed and also inspect
and [alter][248] the response before it's returned to the expectations. You can regard it as an
"around advice" in AOP terms. Filters can be used to implement custom authentication schemes,
session management, logging etc. To create a filter you need to implement the
[io.restassured.filter.Filter][249] interface. To use a filter you can do:

given().filter(new MyFilter()). ..

There are a couple of filters provided by REST Assured that are ready to use:

1. `io.restassured.filter.log.RequestLoggingFilter`: A filter that'll print the request
   specification details.
2. `io.restassured.filter.log.ResponseLoggingFilter`: A filter that'll print the response details if
   the response matches a given status code.
3. `io.restassured.filter.log.ErrorLoggingFilter`: A filter that'll print the response body if an
   error occurred (status code is between 400 and 500).

### Ordered Filters

As of REST Assured 3.0.2 you can implement the [io.restassured.filter.OrderedFilter][250] interface
if you need to control the filter ordering. Here you implement the `getOrder` method to return an
integer representing the precedence of the filter. A lower value gives higher precedence. The
highest precedence you can define is `Integer.MIN_VALUE` and the lowest precedence is
`Integer.MAX_VALUE`. Filters not implementing [io.restassured.filter.OrderedFilter][251] will have a
default precedence of `1000`. Click [here][252] for some examples.

### Response Builder

If you need to change the [Response][253] from a filter you can use the [ResponseBuilder][254] to
create a new Response based on the original response. For example if you want to change the body of
the original response to something else you can do:

Response newResponse = new ResponseBuilder().clone(originalResponse).setBody("Something").build();

# Logging

In many cases it can be useful to print the response and/or request details in order to help you
create the correct expectations and send the correct requests. To do help you do this you can use
one of the predefined [filters][255] supplied with REST Assured or you can use one of the shortcuts.

## Request Logging

Since version 1.5 REST Assured supports logging the *[request specification][256]* before it's sent
to the server using the [RequestLoggingFilter][257]. Note that the HTTP Builder and HTTP Client may
add additional headers then what's printed in the log. The filter will *only* log details specified
in the request specification. I.e. you can NOT regard the details logged by the
[RequestLoggingFilter][258] to be what's actually sent to the server. Also subsequent filters may
alter the request *after* the logging has taken place. If you need to log what's *actually* sent on
the wire refer to the [HTTP Client logging docs][259] or use an external tool such [Wireshark][260].
Examples:

given().log().all(). .. // Log all request specification details including parameters, headers and b
ody
given().log().params(). .. // Log only the parameters of the request
given().log().body(). .. // Log only the request body
given().log().headers(). .. // Log only the request headers
given().log().cookies(). .. // Log only the request cookies
given().log().method(). .. // Log only the request method
given().log().path(). .. // Log only the request path

## Response Logging

If you want to print the response body regardless of the status code you can do:

get("/x").then().log().body() ..

This will print the response body regardless if an error occurred. If you're only interested in
printing the response body if an error occur then you can use:

get("/x").then().log().ifError(). .. 

You can also log all details in the response including status line, headers and cookies:

get("/x").then().log().all(). .. 

as well as only status line, headers or cookies:

get("/x").then().log().statusLine(). .. // Only log the status line
get("/x").then().log().headers(). .. // Only log the response headers
get("/x").then().log().cookies(). .. // Only log the response cookies

You can also configure to log the response only if the status code matches some value:

get("/x").then().log().ifStatusCodeIsEqualTo(302). .. // Only log if the status code is equal to 302
get("/x").then().log().ifStatusCodeMatches(matcher). .. // Only log if the status code matches the s
upplied Hamcrest matcher

## Log if validation fails

Since REST Assured 2.3.1 you can log the request or response only if the validation fails. To log
the request do:

given().log().ifValidationFails(). ..

To log the response do:

.. .then().log().ifValidationFails(). ..

It's also possible to enable this for both the request and the response at the same time using the
[LogConfig][261]:

given().config(RestAssured.config().logConfig(logConfig().enableLoggingOfRequestAndResponseIfValidat
ionFails(HEADERS))). ..

This will log only the headers if validation fails.

There's also a shortcut for enabling logging of the request and response for all requests if
validation fails:

RestAssured.enableLoggingOfRequestAndResponseIfValidationFails();

As of version 4.5.0 you can also use the `onFailMessage` to specify a message that'll been shown
when a test fails. For example:

when().
      get().
then().
      onFailMessage("Some specific message").
      statusCode(200);

Now "Some specific message" will be shown in the error message. This is good if you want to e.g.
distinguish more easily between tests if they fail.

## Blacklist Headers from Logging

As of REST Assured 4.2.0 it's possible to blacklist headers so that they are not shown in the
request or response log. Instead the header value will be replaced with `[ BLACKLISTED ]`. You can
enable this per header basis using the [LogConfig][262]:

given().config(config().logConfig(logConfig().blacklistHeader("Accept"))). ..

The response log will the print:

`Request method:   GET
Request URI:    http://localhost:8080/something
Proxy:          <none>
Request params: <none>
Query params:   <none>
Form params:    <none>
Path params:    <none>
Headers:        Accept=[ BLACKLISTED ]
Cookies:        <none>
Multiparts:     <none>
Body:           <none>
`

# Root path

To avoid duplicated paths in body expectations you can specify a root path. E.g. instead of writing:

when().
         get("/something").
then().
         body("x.y.firstName", is(..)).
         body("x.y.lastName", is(..)).
         body("x.y.age", is(..)).
         body("x.y.gender", is(..));

you can use a root path and do:

when().
        get("/something").
then().
         root("x.y"). // You can also use the "root" method
         body("firstName", is(..)).
         body("lastName", is(..)).
         body("age", is(..)).
         body("gender", is(..));

You can also set a default root path using:

RestAssured.rootPath = "x.y";

In more advanced use cases it may also be useful to append additional root arguments to existing
root arguments. To do this you can use the `appendRoot` method, for example:

when().
         get("/jsonStore").
then().
         root("store.%s", withArgs("book")).
         body("category.size()", equalTo(4)).
         appendRoot("%s.%s", withArgs("author", "size()")).
         body(withNoArgs(), equalTo(4));

It's also possible to detach a root. For example:

when().
         get("/jsonStore").
then().
         root("store.category").
         body("size()", equalTo(4)).
         detachRoot("category").
         body("size()", equalTo(1));

# Path arguments

Path arguments are useful in situations where you have e.g. pre-defined variables that constitutes
the path. For example

String someSubPath = "else";
int index = 1;
get("/x").then().body("something.%s[%d]", withArgs(someSubPath, index), equalTo("some value")). ..

will expect that the body path "`something.else[0]`" is equal to "some value".

Another usage is if you have complex [root paths][263] and don't wish to duplicate the path for
small variations:

when().
       get("/x").
then().
       root("filters.filterConfig[%d].filterConfigGroups.find { it.name == 'GroupName' }.includes").
       body(withArgs(0), hasItem("first")).
       body(withArgs(1), hasItem("second")).
       ..

The path arguments follows the standard [formatting syntax][264] of Java.

Note that the `withArgs` method can be statically imported from the
[io.restassured.RestAssured][265] class.

Sometimes it's also useful to validate a body without any additional arguments when all arguments
have already been specified in the root path. This is where `withNoArgs` come into play. For
example:

when().
         get("/jsonStore").
then().
         root("store.%s", withArgs("book")).
         body("category.size()", equalTo(4)).
         appendRoot("%s.%s", withArgs("author", "size()")).
         body(withNoArgs(), equalTo(4));

# Session support

REST Assured provides a simplified way for managing sessions. You can define a session id value in
the DSL:

given().sessionId("1234"). .. 

This is actually just a short-cut for:

given().cookie("JSESSIONID", "1234"). .. 

You can also specify a default `sessionId` that'll be supplied with all subsequent requests:

RestAssured.sessionId = "1234";

By default the session id name is `JSESSIONID` but you can change it using the [SessionConfig][266]:

RestAssured.config = RestAssured.config().sessionConfig(new SessionConfig().sessionIdName("phpsessio
nid"));

You can also specify a sessionId using the `RequestSpecBuilder` and reuse it in many tests:

RequestSpecBuilder spec = new RequestSpecBuilder().setSessionId("value1").build();
   
// Make the first request with session id equal to value1
given().spec(spec). .. 
// Make the second request with session id equal to value1
given().spec(spec). .. 

It's also possible to get the session id from the response object:

String sessionId = get("/something").sessionId();

## Session Filter

As of version 2.0.0 you can use a [session filter][267] to automatically capture and apply the
session, for example:

SessionFilter sessionFilter = new SessionFilter();

given().
          auth().form("John", "Doe").
          filter(sessionFilter).
when().
          get("/formAuth").
then().
          statusCode(200);


given().
          filter(sessionFilter). // Reuse the same session filter instance to automatically apply th
e session id from the previous response
when().
          get("/x").
then().
          statusCode(200);

To get session id caught by the `SessionFilter` you can do like this:

String sessionId = sessionFilter.getSessionId();

# SSL

In most situations SSL should just work out of the box thanks to the excellent work of HTTP Builder
and HTTP Client. There are however some cases where you'll run into trouble. You may for example run
into a SSLPeerUnverifiedException if the server is using an invalid certificate. The easiest way to
workaround this is to use "relaxed HTTPs validation". For example:

given().relaxedHTTPSValidation().when().get("https://some_server.com"). .. 

You can also define this statically for all requests:

RestAssured.useRelaxedHTTPSValidation();

or in a [request specification][268].

This will assume an SSLContext protocol of `SSL`. To change to another protocol use an overloaded
versionen of `relaxedHTTPSValidation`. For example:

given().relaxedHTTPSValidation("TLS").when().get("https://some_server.com"). .. 

You can also be more fine-grained and create Java keystore file and use it with REST Assured. It's
not too difficult, first follow the guide [here][269] and then use the keystore in Rest Assured like
this:

given().keystore("/pathToJksInClassPath", <password>). .. 

or you can specify it for every request:

RestAssured.keystore("/pathToJksInClassPath", <password>);

You can also define a keystore in a re-usable [specification][270].

If you already loaded a keystore with a password you can use it as a truststore:

RestAssured.trustStore(keystore);

You can find a working example [here][271].

For more advanced SSL Configuration refer to the [SSL Configuration][272] section.

## SSL invalid hostname

If the certificate is specifying an invalid hostname you don't need to create and import a keystore.
As of version `2.2.0` you can do:

RestAssured.config = RestAssured.config().sslConfig(sslConfig().allowAllHostnames());

to allow all hostnames for all requests or:

given().config(RestAssured.config().sslConfig(sslConfig().allowAllHostnames()). .. ;

for a single request.

Note that if you use "relaxed HTTPs validation" then `allowAllHostnames` is activated by default.

# URL Encoding

Usually you don't have to think about URL encoding since Rest Assured provides this automatically
out of the box. In some cases though it may be useful to turn URL Encoding off. One reason may be
that you already the have some parameters encoded before you supply them to Rest Assured. To prevent
double URL encoding you need to tell Rest Assured to disable it's URL encoding. E.g.

String response = given().urlEncodingEnabled(false).get("https://jira.atlassian.com:443/rest/api/2.0
.alpha1/search?jql=project%20=%20BAM%20AND%20issuetype%20=%20Bug").asString();
..

or

RestAssured.baseURI = "https://jira.atlassian.com";
RestAssured.port = 443;
RestAssured.urlEncodingEnabled = false;
final String query = "project%20=%20BAM%20AND%20issuetype%20=%20Bug";
String response = get("/rest/api/2.0.alpha1/search?jql={q}", query);
..

# Proxy Configuration

Starting from version 2.3.2 REST Assured has better support for proxies. For example if you have a
proxy at localhost port 8888 you can do:

given().proxy("localhost", 8888). .. 

Actually you don't even have to specify the hostname if the server is running on your local
environment:

given().proxy(8888). .. // Will assume localhost

To use HTTPS you need to supply a third parameter (scheme) or use the
`io.restassured.specification.ProxySpecification`. For example:

given().proxy(host("localhost").withScheme("https")). ..

where `host` is statically imported from `io.restassured.specification.ProxySpecification`.

Starting from version 2.7.0 you can also specify preemptive basic authentication for proxies. For
example:

`given().proxy(auth("username", "password")).when() ..
`

where `auth` is statically imported from [io.restassured.specification.ProxySpecification][273]. You
can of course also combine authentication with a different host:

given().proxy(host("http://myhost.org").withAuth("username", "password")). ..

## Static Proxy Configuration

It's also possible to configure a proxy statically for all requests, for example:

RestAssured.proxy("localhost", 8888);    

or:

RestAssured.proxy = host("localhost").withPort(8888);

## Request Specification Proxy Configuration

You can also create a request specification and specify the proxy there:

RequestSpecification specification = new RequestSpecBuilder().setProxy("localhost").build();
given().spec(specification). ..

# Detailed configuration

Detailed configuration is provided by the [RestAssuredConfig][274] instance with which you can
configure the parameters of [HTTP Client][275] as well as [Redirect][276], [Log][277],
[Encoder][278], [Decoder][279], [Session][280], [ObjectMapper][281], [Connection][282], [SSL][283]
and [ParamConfig][284] settings. Examples:

For a specific request:

given().config(RestAssured.config().redirect(redirectConfig().followRedirects(false))). ..

or using a RequestSpecBuilder:

RequestSpecification spec = new RequestSpecBuilder().setConfig(RestAssured.config().redirect(redirec
tConfig().followRedirects(false))).build();

or for all requests:

RestAssured.config = config().redirect(redirectConfig().followRedirects(true).and().maxRedirects(0))
;

`config()` and `newConfig()` can be statically imported from
`io.restassured.config.RestAssuredConfig`.

## Encoder Config

With the [EncoderConfig][285] you can specify the default content encoding charset (if it's not
specified in the content-type header) and query parameter charset for all requests. If no content
charset is specified then ISO-8859-1 is used and if no query parameter charset is specified then
UTF-8 is used. Usage example:

RestAssured.config = RestAssured.config().encoderConfig(encoderConfig().defaultContentCharset("US-AS
CII"));

You can also specify which encoder charset to use for a specific content-type if no charset is
defined explicitly for this content-type by using the `defaultCharsetForContentType` method in the
[EncoderConfig][286]. For example:

RestAssured.config = RestAssured.config(config().encoderConfig(encoderConfig().defaultCharsetForCont
entType("UTF-16", "application/xml")));

This will assume UTF-16 encoding for "application/xml" content-types that does explicitly specify a
charset. By default "application/json" is specified to use "UTF-8" as default content-type as this
is specified by [RFC4627][287].

### Avoid adding the charset to content-type header automatically

By default REST Assured adds the charset header automatically. To disable this completely you can
configure the `EncoderConfig` like this:

RestAssured.config = RestAssured.config(config().encoderConfig(encoderConfig().appendDefaultContentC
harsetToContentTypeIfUndefined(false));

## Decoder Config

With the [DecoderConfig][288] you can set the default response content decoding charset for all
responses. This is useful if you expect a different content charset than ISO-8859-1 (which is the
default charset) and the response doesn't define the charset in the content-type header. Usage
example:

RestAssured.config = RestAssured.config().decoderConfig(decoderConfig().defaultContentCharset("UTF-8
"));

You can also use the `DecoderConfig` to specify which content decoders to apply. When you do this
the `Accept-Encoding` header will be added automatically to the request and the response body will
be decoded automatically. By default GZIP and DEFLATE decoders are enabled. To for example to remove
GZIP decoding but retain DEFLATE decoding you can do the following:

given().config(RestAssured.config().decoderConfig(decoderConfig().contentDecoders(DEFLATE))). ..

You can also specify which decoder charset to use for a specific content-type if no charset is
defined explicitly for this content-type by using the "defaultCharsetForContentType" method in the
[DecoderConfig][289]. For example:

RestAssured.config = config(config().decoderConfig(decoderConfig().defaultCharsetForContentType("UTF
-16", "application/xml")));

This will assume UTF-16 encoding for "application/xml" content-types that does explicitly specify a
charset. By default "application/json" is using "UTF-8" as default charset as this is specified by
[RFC4627][290].

## Session Config

With the session config you can configure the default session id name that's used by REST Assured.
The default session id name is `JSESSIONID` and you only need to change it if the name in your
application is different and you want to make use of REST Assured's [session support][291]. Usage:

RestAssured.config = RestAssured.config().sessionConfig(new SessionConfig().sessionIdName("phpsessio
nid"));

## Redirect DSL

Redirect configuration can also be specified using the DSL. E.g.

given().redirects().max(12).and().redirects().follow(true).when(). .. 

## Connection Config

Lets you configure connection settings for REST Assured. For example if you want to force-close the
Apache HTTP Client connection after each response. You may want to do this if you make a lot of fast
consecutive requests with small amount of data in the response. However if you're downloading
(especially large amounts of) chunked data you must not close connections after each response. By
default connections are *not* closed after each response.

RestAssured.config = RestAssured.config().connectionConfig(connectionConfig().closeIdleConnectionsAf
terEachResponse());

## Json Config

[JsonPathConfig][292] allows you to configure the Json settings either when used by REST Assured or
by [JsonPath][293]. It let's you configure how JSON numbers should be treated.

RestAssured.config = RestAssured.config().jsonConfig(jsonConfig().numberReturnType(NumberReturnType.
BIG_DECIMAL))

## HTTP Client Config

Let's you configure properties for the HTTP Client instance that REST Assured will be using when
executing requests. By default REST Assured creates a new instance of http client for each "given"
statement. To configure reuse do the following:

RestAssured.config = RestAssured.config().httpClient(httpClientConfig().reuseHttpClientInstance());

You can also supply a custom HTTP Client instance by using the `httpClientFactory` method, for
example:

RestAssured.config = RestAssured.config().httpClient(httpClientConfig().httpClientFactory(
         new HttpClientConfig.HttpClientFactory() {

            @Override
            public HttpClient createHttpClient() {
                return new SystemDefaultHttpClient();
            }
        }));

**Note that currently you need to supply an instance of `AbstractHttpClient`.**

It's also possible to configure default parameters etc.

## SSL Config

The [SSLConfig][294] allows you to specify more advanced SSL configuration such as truststore,
keystore type and host name verifier. For example:

RestAssured.config = RestAssured.config().sslConfig(sslConfig().with().keystoreType(<type>).and().st
rictHostnames());

## Param Config

[ParamConfig][295] allows you to configure how different parameter types should be updated on
"collision". By default all parameters are merged so if you do:

given().queryParam("param1", "value1").queryParam("param1", "value2").when().get("/x"). ...

REST Assured will send a query string of `param1=value1&param1=value2`. This is not always what you
want though so you can configure REST Assured to *replace* values instead:

given().
        config(config().paramConfig(paramConfig().queryParamsUpdateStrategy(REPLACE))).
        queryParam("param1", "value1").
        queryParam("param1", "value2").
when().
        get("/x"). ..

REST Assured will now replace `param1` with `value2` (since it's written last) instead of merging
them together. You can also configure the update strategy for each type of for all parameter types
instead of doing it per individual basis:

given().config(config().paramConfig(paramConfig().replaceAllParameters())). ..

This is also supported in the [Spring Mock Mvc Module][296] (but the config there is called
[MockMvcParamConfig][297]).

## Failure Config

Added in version 3.3.0 the [FailureConfig][298] can be used to get callbacks when REST Assured
validation fails. This is useful if you want to do some custom logging or store data available in
the request/response specification or in the response itself somewhere. For example let's say that
you want to be notified by email when the following test case fails because the status code is not
200:

given().
    param("x", "y")
when().
    get("/hello")
then().
    statusCode(200);

You can then implement a [ResponseValidationFailureListener][299] and add it to the
[FailureConfig][300]:

ResponseValidationFailureListener emailOnFailure = (reqSpec, respSpec, resp) -> emailService.sendEma
il("email@gmail.com", "Important test failed! Status code was: " + resp.statusCode());

given().
    config(RestAssured.config().failureConfig(failureConfig().with().failureListeners(emailOnFailure
))).
    param("x", "y")
when().
    get("/hello")
then().
    statusCode(200);

## CSRF Config

Added in version 5.2.0 the [CsrfConfig][301] can be used to configure REST Assured to support CSRF.
It provides more fine-grained control that is not available in the normal [DSL][302].

given().
    config(config().csrfConfig(csrfConfig().with().csrfTokenPath("/loginPageWithCsrf").and().logging
Enabled(LogDetail.BODY))).
when().
    post("/loginPageWithCsrf").
then().
    statusCode(200);

# More info

For more information refer to the [javadoc][303]:

* [RestAssured][304]
* [RestAssuredMockMvc Javadoc][305]
* [Specification package][306]

You can also have a look at some code examples:

* REST Assured [tests][307]
* [JsonPathTest][308]
* [XmlPathTest][309]

If you need support then join the [mailing list][310].

For professional support please contact [johanhaleby][311].

## Toggle table of contents Pages 40

* Loading [
  Home
  ][312]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][313].
* Loading [
  Downloads
  ][314]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][315].
* Loading [
  FAQ
  ][316]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][317].
* Loading [
  GettingStarted
  ][318]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][319].
* Loading [
  How_to_release
  ][320]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][321].
* Loading [
  Kotlin
  ][322]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][323].
* Loading [
  OldNews
  ][324]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][325].
* Loading [
  ReleaseNotes
  ][326]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][327].
* Loading [
  ReleaseNotes11
  ][328]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][329].
* Loading [
  ReleaseNotes145
  ][330]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][331].
* Loading [
  ReleaseNotes15
  ][332]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][333].
* Loading [
  ReleaseNotes16
  ][334]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][335].
* Loading [
  ReleaseNotes17
  ][336]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][337].
* Loading [
  ReleaseNotes18
  ][338]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][339].
* Loading [
  ReleaseNotes19
  ][340]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][341].
* Loading [
  ReleaseNotes20
  ][342]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][343].
* Loading [
  ReleaseNotes21
  ][344]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][345].
* Loading [
  ReleaseNotes22
  ][346]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][347].
* Loading [
  ReleaseNotes23
  ][348]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][349].
* Loading [
  ReleaseNotes24
  ][350]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][351].
* Loading [
  ReleaseNotes25
  ][352]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][353].
* Loading [
  ReleaseNotes26
  ][354]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][355].
* Loading [
  ReleaseNotes27
  ][356]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][357].
* Loading [
  ReleaseNotes28
  ][358]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][359].
* Loading [
  ReleaseNotes29
  ][360]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][361].
* Loading [
  ReleaseNotes30
  ][362]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][363].
* Loading [
  ReleaseNotes31
  ][364]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][365].
* Loading [
  ReleaseNotes32
  ][366]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][367].
* Loading [
  ReleaseNotes33
  ][368]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][369].
* Loading [
  ReleaseNotes40
  ][370]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][371].
* Loading [
  ReleaseNotes42
  ][372]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][373].
* Loading [
  ReleaseNotes45
  ][374]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][375].
* Loading [
  ReleaseNotes50
  ][376]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][377].
* Loading [
  ReleaseNotes52
  ][378]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][379].
* Loading [
  ReleaseNotes60
  ][380]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][381].
* Loading [
  Scala
  ][382]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][383].
* Loading [
  snapshot
  ][384]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][385].
* Loading [
  Spring
  ][386]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][387].
* Loading [
  Usage
  ][388]
  
  * [Contents][389]
  * [Static imports][390]
  * [Examples][391]
  * [Example 1 - JSON][392]
  * [Returning floats and doubles as BigDecimal][393]
  * [JSON Schema validation][394]
  * [JSON Schema Validation Settings][395]
  * [Json Schema Validation with static configuration][396]
  * [Json Schema Validation without REST Assured][397]
  * [Anonymous JSON root validation][398]
  * [Example 2 - XML][399]
  * [XML namespaces][400]
  * [XPath][401]
  * [Schema and DTD validation][402]
  * [XSD example][403]
  * [DTD example][404]
  * [Example 3 - Complex parsing and validation][405]
  * [XML Example][406]
  * [Depth-first search][407]
  * [JSON Example][408]
  * [Example 1][409]
  * [Example 2][410]
  * [Deserialization with Generics][411]
  * [Additional Examples][412]
  * [Note on floats and doubles][413]
  * [Note on syntax][414]
  * [Syntactic Sugar][415]
  * [Getting Response Data][416]
  * [Extracting values from the Response after validation][417]
  * [JSON (using JsonPath)][418]
  * [JsonPath Configuration][419]
  * [XML (using XmlPath)][420]
  * [XmlPath Configuration][421]
  * [Parsing HTML with XmlPath][422]
  * [Single path][423]
  * [Headers, cookies, status etc][424]
  * [Multi-value headers and cookies][425]
  * [Multi-value headers][426]
  * [Multi-value cookies][427]
  * [Detailed Cookies][428]
  * [Specifying Request Data][429]
  * [Invoking HTTP resources][430]
  * [Parameters][431]
  * [Multi-value parameter][432]
  * [No-value parameter][433]
  * [Path parameters][434]
  * [Cookies][435]
  * [Headers][436]
  * [Header Merging/Overwriting][437]
  * [Content Type][438]
  * [Request Body][439]
  * [Verifying Response Data][440]
  * [Response Body][441]
  * [Cookies][442]
  * [Status][443]
  * [Headers][444]
  * [Content-Type][445]
  * [Full body/content matching][446]
  * [Use the response to verify other parts of the response][447]
  * [Measuring Response Time][448]
  * [Authentication][449]
  * [Basic Authentication][450]
  * [Preemptive Basic Authentication][451]
  * [Challenged Basic Authentication][452]
  * [Digest Authentication][453]
  * [Form Authentication][454]
  * [CSRF][455]
  * [CSRF Form Token][456]
  * [CSRF Header Token][457]
  * [CSRF and Authentication][458]
  * [CSRF Prioritization][459]
  * [CSRF Cookie Propagation][460]
  * [OAuth][461]
  * [OAuth 1][462]
  * [OAuth 2][463]
  * [Custom Authentication][464]
  * [Multi-part form data][465]
  * [Object Mapping][466]
  * [Jakarta EE][467]
  * [JAXB][468]
  * [Serialization][469]
  * [Content-Type based Serialization][470]
  * [Create JSON from a HashMap][471]
  * [Using an Explicit Serializer][472]
  * [Deserialization][473]
  * [Content-Type based Deserialization][474]
  * [Custom Content-Type Deserialization][475]
  * [Using an Explicit Deserializer][476]
  * [Configuration][477]
  * [Custom][478]
  * [Custom parsers][479]
  * [Default parser][480]
  * [Default values][481]
  * [Specification Re-use][482]
  * [Querying RequestSpecification][483]
  * [Filters][484]
  * [Ordered Filters][485]
  * [Response Builder][486]
  * [Logging][487]
  * [Request Logging][488]
  * [Response Logging][489]
  * [Log if validation fails][490]
  * [Blacklist Headers from Logging][491]
  * [Root path][492]
  * [Path arguments][493]
  * [Session support][494]
  * [Session Filter][495]
  * [SSL][496]
  * [SSL invalid hostname][497]
  * [URL Encoding][498]
  * [Proxy Configuration][499]
  * [Static Proxy Configuration][500]
  * [Request Specification Proxy Configuration][501]
  * [Detailed configuration][502]
  * [Encoder Config][503]
  * [Avoid adding the charset to content-type header automatically][504]
  * [Decoder Config][505]
  * [Session Config][506]
  * [Redirect DSL][507]
  * [Connection Config][508]
  * [Json Config][509]
  * [HTTP Client Config][510]
  * [SSL Config][511]
  * [Param Config][512]
  * [Failure Config][513]
  * [CSRF Config][514]
  * [More info][515]
* Loading [
  Usage_Legacy
  ][516]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][517].
* Show 25 more pages…

* [Getting Started][518]
* [Downloads][519]
* [Usage Guide][520] ([Legacy][521])
* [Snapshot dependencies][522]
* [Release Notes][523]
* [FAQ][524]
* [Support][525]

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
[25]: /rest-assured/rest-assured/wiki/Usage/_history
[26]: Usage_Legacy
[27]: #static-imports
[28]: #examples
[29]: #example-1---json
[30]: #json-schema-validation
[31]: #example-2---xml
[32]: #example-3---complex-parsing-and-validation
[33]: #xml-example
[34]: #json-example
[35]: #deserialization-with-generics
[36]: #additional-examples
[37]: #note-on-floats-and-doubles
[38]: #note-on-syntax
[39]: #syntactic-sugar
[40]: #getting-response-data
[41]: #extracting-values-from-the-response-after-validation
[42]: #json-using-jsonpath
[43]: #xml-using-xmlpath
[44]: #single-path
[45]: #headers-cookies-status-etc
[46]: #multi-value-headers
[47]: #multi-value-cookies
[48]: #detailed-cookies
[49]: #specifying-request-data
[50]: #invoking-http-resources
[51]: #parameters
[52]: #multi-value-parameter
[53]: #no-value-parameter
[54]: #path-parameters
[55]: #cookies
[56]: #headers
[57]: #content-type
[58]: #request-body
[59]: #verifying-response-data
[60]: #response-body
[61]: #cookies-1
[62]: #status
[63]: #headers-1
[64]: #content-type-1
[65]: #full-bodycontent-matching
[66]: #use-the-response-to-verify-other-parts-of-the-response
[67]: #measuring-response-time
[68]: #authentication
[69]: #basic-authentication
[70]: #preemptive-basic-authentication
[71]: #challenged-basic-authentication
[72]: #digest-authentication
[73]: #form-authentication
[74]: #oauth
[75]: #oauth-1
[76]: #oauth-2
[77]: #csrf
[78]: #csrf-form-token
[79]: #csrf-header-token
[80]: #csrf-and-authentication
[81]: #csrf-prioritization
[82]: #csrf-cookie-propagation
[83]: #multi-part-form-data
[84]: #object-mapping
[85]: #jakarta-ee
[86]: #jaxb
[87]: #serialization
[88]: #content-type-based-serialization
[89]: #create-json-from-a-hashmap
[90]: #using-an-explicit-serializer
[91]: #deserialization
[92]: #content-type-based-deserialization
[93]: #custom-content-type-deserialization
[94]: #using-an-explicit-deserializer
[95]: #configuration
[96]: #custom
[97]: #custom-parsers
[98]: #default-parser
[99]: #default-values
[100]: #specification-re-use
[101]: #querying-requestspecification
[102]: #filters
[103]: #ordered-filters
[104]: #response-builder
[105]: #logging
[106]: #request-logging
[107]: #response-logging
[108]: #log-if-validation-fails
[109]: #blacklist-headers-from-logging
[110]: #root-path
[111]: #path-arguments
[112]: #session-support
[113]: #session-filter
[114]: #ssl
[115]: #ssl-invalid-hostname
[116]: #url-encoding
[117]: #proxy-configuration
[118]: #static-proxy-configuration
[119]: #request-specification-proxy-configuration
[120]: #detailed-configuration
[121]: #encoder-config
[122]: #decoder-config
[123]: #session-config
[124]: #redirect-dsl
[125]: #connection-config
[126]: #json-config
[127]: #http-client-config
[128]: #ssl-config
[129]: #param-config
[130]: #failure-config
[131]: #csrf-config
[132]: Spring#spring-support
[133]: Spring#spring-mock-mvc-module
[134]: Spring#bootstrapping-restassuredmockmvc
[135]: Spring#asynchronous-requests
[136]: Spring#adding-request-post-processors
[137]: Spring#adding-result-handlers
[138]: Spring#using-result-matchers
[139]: Spring#interceptors
[140]: Spring#specifications
[141]: Spring#resetting-restassuredmockmvc
[142]: Spring#spring-mvc-authentication
[143]: Spring#using-spring-security-test
[144]: Spring#injecting-a-user
[145]: Spring#spring-web-test-client-module
[146]: Spring#bootstrapping-restassuredwebtestclient
[147]: Spring#spring-web-test-client-specifications
[148]: Spring#resetting-restassuredwebtestclient
[149]: Kotlin#kotlin-extension-module-for-spring-webtest
[150]: Spring#common-spring-module-documentation
[151]: Spring#note-on-parameters
[152]: Scala#scala
[153]: Scala#scala-extension-module
[154]: Scala#scala-support-module
[155]: Kotlin#kotlin
[156]: Kotlin#avoid-escaping-when-keyword
[157]: Kotlin#kotlin-extension-module
[158]: Kotlin#kotlin-extension-module-for-spring-mockmvc
[159]: Kotlin#kotlin-extension-module-for-spring-webtest
[160]: #more-info
[161]: http://json-schema.org/
[162]: #json-schema-validation
[163]: #spring-mock-mvc-module
[164]: http://static.javadoc.io/io.restassured/spring-mock-mvc/6.0.0/io/restassured/module/mockmvc/R
estAssuredMockMvc.html
[165]: http://localhost:8080/lotto
[166]: http://groovy-lang.org/processing-xml.html#_gpath
[167]: https://github.com/jayway/JsonPath
[168]: http://json-schema.org/
[169]: http://dl.bintray.com/johanhaleby/generic/json-schema-validator-6.0.0-dist.zip
[170]: https://github.com/fge/json-schema-validator
[171]: http://static.javadoc.io/io.restassured/json-schema-validator/6.0.0/io/restassured/module/jsv
/JsonSchemaValidatorSettings.html
[172]: http://static.javadoc.io/io.restassured/json-schema-validator/6.0.0/io/restassured/module/jsv
/JsonSchemaValidatorSettings.html
[173]: http://static.javadoc.io/io.restassured/json-schema-validator/6.0.0/io/restassured/module/jsv
/JsonSchemaValidatorSettings.html
[174]: GattingStarted
[175]: http://groovy-lang.org/processing-xml.html#_gpath
[176]: http://groovy-lang.org/processing-xml.html#_gpath
[177]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/XmlConfig.h
tml
[178]: https://github.com/rest-assured/rest-assured/wiki/ReleaseNotes26#non-backward-compatible-chan
ges
[179]: http://docs.oracle.com/javase/7/docs/api/javax/xml/namespace/NamespaceContext.html
[180]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/matcher/RestAssure
dMatchers.html
[181]: http://static.javadoc.io/io.restassured/xml-path/6.0.0/io/restassured/path/xml/XmlPath.html
[182]: #single-path
[183]: http://static.javadoc.io/io.restassured/json-path/6.0.0/io/restassured/path/json/JsonPath.htm
l
[184]: #example-3---complex-parsing-and-validation
[185]: http://static.javadoc.io/io.restassured/json-path/6.0.0/io/restassured/path/json/JsonPath.htm
l
[186]: https://www.hascode.com/testing-restful-web-services-made-easy-using-the-rest-assured-framewo
rk/
[187]: https://www.linkedin.com/in/basdijkstra
[188]: http://www.ontestautomation.com/open-sourcing-my-workshop-an-experiment/
[189]: https://github.com/basdijkstra/workshops/
[190]: https://testautomationu.applitools.com/automating-your-api-tests-with-rest-assured/
[191]: http://static.javadoc.io/io.restassured/json-path/6.0.0/io/restassured/path/json/JsonPath.htm
l
[192]: GettingStarted
[193]: http://www.jayway.com/2013/04/12/whats-new-in-rest-assured-1-8/
[194]: http://groovy-lang.org/processing-xml.html#_gpath
[195]: https://github.com/jayway/JsonPath
[196]: http://static.javadoc.io/io.restassured/xml-path/6.0.0/io/restassured/path/xml/XmlPath.html
[197]: GettingStarted
[198]: http://www.jayway.com/2013/04/12/whats-new-in-rest-assured-1-8/
[199]: http://static.javadoc.io/io.rest-assured/xml-path/6.0.0/io/restassured/path/xml/XmlPath.Compa
tibilityMode.html
[200]: #default-parser
[201]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/http/Headers.html
[202]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/response/Response.
html
[203]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/http/Headers.html#
getValues(java.lang.String)
[204]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/http/Cookies.html
[205]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/response/Response.
html
[206]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/http/Cookies.html#
getValues(java.lang.String)
[207]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/http/Cookie.html
[208]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/response/ResponseO
ptions.html#getDetailedCookie-java.lang.String-
[209]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/http/Cookies.html
[210]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/response/ResponseO
ptions.html#getDetailedCookies--
[211]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/specification/Requ
estSpecification.html
[212]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/specification/Requ
estSpecification.html#request-java.lang.String-java.lang.String-
[213]: #multi-part-form-data
[214]: #filters
[215]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/HeaderConfi
g.html
[216]: #serialization
[217]: #example-1---json
[218]: #example-2---xml
[219]: #deserialization
[220]: #specification-re-use
[221]: https://en.wikipedia.org/wiki/Form-based_authentication
[222]: https://en.wikipedia.org/wiki/Cross-site_request_forgery
[223]: #session-filter
[224]: https://github.com/fernandezpablo85/scribe-java
[225]: Usage_Legacy#OAuth
[226]: https://github.com/fernandezpablo85/scribe-java/releases
[227]: #oauth
[228]: #oauth
[229]: #oauth
[230]: #filters
[231]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/builder/MultiPartS
pecBuilder.html
[232]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/MultiPartCo
nfig.html
[233]: http://blog.jayway.com/2011/09/15/multipart-form-data-file-uploading-made-simple-with-rest-as
sured/
[234]: #jakarta-ee
[235]: http://code.google.com/p/rest-assured/wiki/Usage#Using_an_Explicit_Deserializer
[236]: #default-values
[237]: #specification-re-use
[238]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/ObjectMappe
rConfig.html
[239]: #detailed-configuration
[240]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/mapper/ObjectMappe
r.html
[241]: https://github.com/rest-assured/rest-assured/blob/master/examples/rest-assured-itest-java/src
/test/java/io/restassured/itest/java/CustomObjectMappingITest.java
[242]: sSpecification-re-use
[243]: #specification-re-use
[244]: http://myhost.org:80/resource/hello
[245]: http://code.google.com/p/rest-assured/wiki/Usage#Root_path
[246]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/builder/RequestSpe
cBuilder.html
[247]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/builder/ResponseSp
ecBuilder.html
[248]: #response-builder
[249]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/filter/Filter.html
[250]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/filter/OrderedFilt
er.html
[251]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/filter/OrderedFilt
er.html
[252]: https://github.com/rest-assured/rest-assured/blob/master/examples/rest-assured-itest-java/src
/test/java/io/restassured/itest/java/OrderedFilterITest.java
[253]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/response/Response.
html
[254]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/builder/ResponseBu
ilder.html
[255]: #filters
[256]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/specification/Requ
estSpecification.html
[257]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/filter/log/Request
LoggingFilter.html
[258]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/filter/log/Request
LoggingFilter.html
[259]: http://hc.apache.org/httpcomponents-client-ga/logging.html
[260]: http://www.wireshark.org/
[261]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/LogConfig.h
tml
[262]: https://www.javadoc.io/doc/io.rest-assured/rest-assured/latest/io/restassured/config/LogConfi
g.html
[263]: http://code.google.com/p/rest-assured/wiki/Usage#Root_path
[264]: http://download.oracle.com/javase/1,5.0/docs/api/java/util/Formatter.html#syntax
[265]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/RestAssured.html
[266]: #session-config
[267]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/filter/session/Ses
sionFilter.html
[268]: #specification-re-use
[269]: https://github.com/jgritman/httpbuilder/wiki/SSL
[270]: http://code.google.com/p/rest-assured/wiki/Usage#Specification_Re-use
[271]: https://github.com/rest-assured/rest-assured/blob/master/examples/rest-assured-itest-java/src
/test/java/io/restassured/itest/java/SSLTest.java
[272]: #ssl-config
[273]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/specification/Prox
ySpecification.html
[274]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/RestAssured
Config.html
[275]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/HttpClientC
onfig.html
[276]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/RedirectCon
fig.html
[277]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/LogConfig.h
tml
[278]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/EncoderConf
ig.html
[279]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/DecoderConf
ig.html
[280]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/SessionConf
ig.html
[281]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/ObjectMappe
rConfig.html
[282]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/ConnectionC
onfig.html
[283]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/SSLConfig.h
tml
[284]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/ParamConfig
.html
[285]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/EncoderConf
ig.html
[286]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/EncoderConf
ig.html
[287]: https://www.ietf.org/rfc/rfc4627.txt
[288]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/DecoderConf
ig.html
[289]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/DecoderConf
ig.html
[290]: https://www.ietf.org/rfc/rfc4627.txt
[291]: #Session_support
[292]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/path/json/config/J
sonPathConfig.html
[293]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/path/json/JsonPath
.html
[294]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/SSLConfig.h
tml
[295]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/ParamConfig
.html
[296]: #spring-mock-mvc-module
[297]: http://static.javadoc.io/io.restassured/spring-mock-mvc/6.0.0/io/restassured/module/mockmvc/c
onfig/MockMvcParamConfig.html
[298]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/FailureConf
ig.html
[299]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/listener/ResponseV
alidationFailureListener.html
[300]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/FailureConf
ig.html
[301]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/config/CsrfConfig.
html
[302]: #csrf
[303]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/index.html
[304]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/RestAssured.html
[305]: http://static.javadoc.io/io.restassured/spring-mock-mvc/6.0.0/io/restassured/module/mockmvc/R
estAssuredMockMvc.html
[306]: http://static.javadoc.io/io.rest-assured/rest-assured/6.0.0/io/restassured/specification/pack
age-summary.html
[307]: https://github.com/rest-assured/rest-assured/tree/master/examples/rest-assured-itest-java/src
/test/java/io/restassured/itest/java
[308]: https://github.com/rest-assured/rest-assured/blob/master/json-path/src/test/java/io/restassur
ed/path/json/JsonPathTest.java
[309]: https://github.com/rest-assured/rest-assured/blob/master/xml-path/src/test/java/io/restassure
d/path/xml/XmlPathTest.java
[310]: http://groups.google.com/group/rest-assured
[311]: https://github.com/johanhaleby
[312]: /rest-assured/rest-assured/wiki
[313]: 
[314]: /rest-assured/rest-assured/wiki/Downloads
[315]: 
[316]: /rest-assured/rest-assured/wiki/FAQ
[317]: 
[318]: /rest-assured/rest-assured/wiki/GettingStarted
[319]: 
[320]: /rest-assured/rest-assured/wiki/How_to_release
[321]: 
[322]: /rest-assured/rest-assured/wiki/Kotlin
[323]: 
[324]: /rest-assured/rest-assured/wiki/OldNews
[325]: 
[326]: /rest-assured/rest-assured/wiki/ReleaseNotes
[327]: 
[328]: /rest-assured/rest-assured/wiki/ReleaseNotes11
[329]: 
[330]: /rest-assured/rest-assured/wiki/ReleaseNotes145
[331]: 
[332]: /rest-assured/rest-assured/wiki/ReleaseNotes15
[333]: 
[334]: /rest-assured/rest-assured/wiki/ReleaseNotes16
[335]: 
[336]: /rest-assured/rest-assured/wiki/ReleaseNotes17
[337]: 
[338]: /rest-assured/rest-assured/wiki/ReleaseNotes18
[339]: 
[340]: /rest-assured/rest-assured/wiki/ReleaseNotes19
[341]: 
[342]: /rest-assured/rest-assured/wiki/ReleaseNotes20
[343]: 
[344]: /rest-assured/rest-assured/wiki/ReleaseNotes21
[345]: 
[346]: /rest-assured/rest-assured/wiki/ReleaseNotes22
[347]: 
[348]: /rest-assured/rest-assured/wiki/ReleaseNotes23
[349]: 
[350]: /rest-assured/rest-assured/wiki/ReleaseNotes24
[351]: 
[352]: /rest-assured/rest-assured/wiki/ReleaseNotes25
[353]: 
[354]: /rest-assured/rest-assured/wiki/ReleaseNotes26
[355]: 
[356]: /rest-assured/rest-assured/wiki/ReleaseNotes27
[357]: 
[358]: /rest-assured/rest-assured/wiki/ReleaseNotes28
[359]: 
[360]: /rest-assured/rest-assured/wiki/ReleaseNotes29
[361]: 
[362]: /rest-assured/rest-assured/wiki/ReleaseNotes30
[363]: 
[364]: /rest-assured/rest-assured/wiki/ReleaseNotes31
[365]: 
[366]: /rest-assured/rest-assured/wiki/ReleaseNotes32
[367]: 
[368]: /rest-assured/rest-assured/wiki/ReleaseNotes33
[369]: 
[370]: /rest-assured/rest-assured/wiki/ReleaseNotes40
[371]: 
[372]: /rest-assured/rest-assured/wiki/ReleaseNotes42
[373]: 
[374]: /rest-assured/rest-assured/wiki/ReleaseNotes45
[375]: 
[376]: /rest-assured/rest-assured/wiki/ReleaseNotes50
[377]: 
[378]: /rest-assured/rest-assured/wiki/ReleaseNotes52
[379]: 
[380]: /rest-assured/rest-assured/wiki/ReleaseNotes60
[381]: 
[382]: /rest-assured/rest-assured/wiki/Scala
[383]: 
[384]: /rest-assured/rest-assured/wiki/snapshot
[385]: 
[386]: /rest-assured/rest-assured/wiki/Spring
[387]: 
[388]: /rest-assured/rest-assured/wiki/Usage
[389]: /rest-assured/rest-assured/wiki/Usage#contents
[390]: /rest-assured/rest-assured/wiki/Usage#static-imports
[391]: /rest-assured/rest-assured/wiki/Usage#examples
[392]: /rest-assured/rest-assured/wiki/Usage#example-1---json
[393]: /rest-assured/rest-assured/wiki/Usage#returning-floats-and-doubles-as-bigdecimal
[394]: /rest-assured/rest-assured/wiki/Usage#json-schema-validation
[395]: /rest-assured/rest-assured/wiki/Usage#json-schema-validation-settings
[396]: /rest-assured/rest-assured/wiki/Usage#json-schema-validation-with-static-configuration
[397]: /rest-assured/rest-assured/wiki/Usage#json-schema-validation-without-rest-assured
[398]: /rest-assured/rest-assured/wiki/Usage#anonymous-json-root-validation
[399]: /rest-assured/rest-assured/wiki/Usage#example-2---xml
[400]: /rest-assured/rest-assured/wiki/Usage#xml-namespaces
[401]: /rest-assured/rest-assured/wiki/Usage#xpath
[402]: /rest-assured/rest-assured/wiki/Usage#schema-and-dtd-validation
[403]: /rest-assured/rest-assured/wiki/Usage#xsd-example
[404]: /rest-assured/rest-assured/wiki/Usage#dtd-example
[405]: /rest-assured/rest-assured/wiki/Usage#example-3---complex-parsing-and-validation
[406]: /rest-assured/rest-assured/wiki/Usage#xml-example
[407]: /rest-assured/rest-assured/wiki/Usage#depth-first-search
[408]: /rest-assured/rest-assured/wiki/Usage#json-example
[409]: /rest-assured/rest-assured/wiki/Usage#example-1
[410]: /rest-assured/rest-assured/wiki/Usage#example-2
[411]: /rest-assured/rest-assured/wiki/Usage#deserialization-with-generics
[412]: /rest-assured/rest-assured/wiki/Usage#additional-examples
[413]: /rest-assured/rest-assured/wiki/Usage#note-on-floats-and-doubles
[414]: /rest-assured/rest-assured/wiki/Usage#note-on-syntax
[415]: /rest-assured/rest-assured/wiki/Usage#syntactic-sugar
[416]: /rest-assured/rest-assured/wiki/Usage#getting-response-data
[417]: /rest-assured/rest-assured/wiki/Usage#extracting-values-from-the-response-after-validation
[418]: /rest-assured/rest-assured/wiki/Usage#json-using-jsonpath
[419]: /rest-assured/rest-assured/wiki/Usage#jsonpath-configuration
[420]: /rest-assured/rest-assured/wiki/Usage#xml-using-xmlpath
[421]: /rest-assured/rest-assured/wiki/Usage#xmlpath-configuration
[422]: /rest-assured/rest-assured/wiki/Usage#parsing-html-with-xmlpath
[423]: /rest-assured/rest-assured/wiki/Usage#single-path
[424]: /rest-assured/rest-assured/wiki/Usage#headers-cookies-status-etc
[425]: /rest-assured/rest-assured/wiki/Usage#multi-value-headers-and-cookies
[426]: /rest-assured/rest-assured/wiki/Usage#multi-value-headers
[427]: /rest-assured/rest-assured/wiki/Usage#multi-value-cookies
[428]: /rest-assured/rest-assured/wiki/Usage#detailed-cookies
[429]: /rest-assured/rest-assured/wiki/Usage#specifying-request-data
[430]: /rest-assured/rest-assured/wiki/Usage#invoking-http-resources
[431]: /rest-assured/rest-assured/wiki/Usage#parameters
[432]: /rest-assured/rest-assured/wiki/Usage#multi-value-parameter
[433]: /rest-assured/rest-assured/wiki/Usage#no-value-parameter
[434]: /rest-assured/rest-assured/wiki/Usage#path-parameters
[435]: /rest-assured/rest-assured/wiki/Usage#cookies
[436]: /rest-assured/rest-assured/wiki/Usage#headers
[437]: /rest-assured/rest-assured/wiki/Usage#header-mergingoverwriting
[438]: /rest-assured/rest-assured/wiki/Usage#content-type
[439]: /rest-assured/rest-assured/wiki/Usage#request-body
[440]: /rest-assured/rest-assured/wiki/Usage#verifying-response-data
[441]: /rest-assured/rest-assured/wiki/Usage#response-body
[442]: /rest-assured/rest-assured/wiki/Usage#cookies-1
[443]: /rest-assured/rest-assured/wiki/Usage#status
[444]: /rest-assured/rest-assured/wiki/Usage#headers-1
[445]: /rest-assured/rest-assured/wiki/Usage#content-type-1
[446]: /rest-assured/rest-assured/wiki/Usage#full-bodycontent-matching
[447]: /rest-assured/rest-assured/wiki/Usage#use-the-response-to-verify-other-parts-of-the-response
[448]: /rest-assured/rest-assured/wiki/Usage#measuring-response-time
[449]: /rest-assured/rest-assured/wiki/Usage#authentication
[450]: /rest-assured/rest-assured/wiki/Usage#basic-authentication
[451]: /rest-assured/rest-assured/wiki/Usage#preemptive-basic-authentication
[452]: /rest-assured/rest-assured/wiki/Usage#challenged-basic-authentication
[453]: /rest-assured/rest-assured/wiki/Usage#digest-authentication
[454]: /rest-assured/rest-assured/wiki/Usage#form-authentication
[455]: /rest-assured/rest-assured/wiki/Usage#csrf
[456]: /rest-assured/rest-assured/wiki/Usage#csrf-form-token
[457]: /rest-assured/rest-assured/wiki/Usage#csrf-header-token
[458]: /rest-assured/rest-assured/wiki/Usage#csrf-and-authentication
[459]: /rest-assured/rest-assured/wiki/Usage#csrf-prioritization
[460]: /rest-assured/rest-assured/wiki/Usage#csrf-cookie-propagation
[461]: /rest-assured/rest-assured/wiki/Usage#oauth
[462]: /rest-assured/rest-assured/wiki/Usage#oauth-1
[463]: /rest-assured/rest-assured/wiki/Usage#oauth-2
[464]: /rest-assured/rest-assured/wiki/Usage#custom-authentication
[465]: /rest-assured/rest-assured/wiki/Usage#multi-part-form-data
[466]: /rest-assured/rest-assured/wiki/Usage#object-mapping
[467]: /rest-assured/rest-assured/wiki/Usage#jakarta-ee
[468]: /rest-assured/rest-assured/wiki/Usage#jaxb
[469]: /rest-assured/rest-assured/wiki/Usage#serialization
[470]: /rest-assured/rest-assured/wiki/Usage#content-type-based-serialization
[471]: /rest-assured/rest-assured/wiki/Usage#create-json-from-a-hashmap
[472]: /rest-assured/rest-assured/wiki/Usage#using-an-explicit-serializer
[473]: /rest-assured/rest-assured/wiki/Usage#deserialization
[474]: /rest-assured/rest-assured/wiki/Usage#content-type-based-deserialization
[475]: /rest-assured/rest-assured/wiki/Usage#custom-content-type-deserialization
[476]: /rest-assured/rest-assured/wiki/Usage#using-an-explicit-deserializer
[477]: /rest-assured/rest-assured/wiki/Usage#configuration
[478]: /rest-assured/rest-assured/wiki/Usage#custom
[479]: /rest-assured/rest-assured/wiki/Usage#custom-parsers
[480]: /rest-assured/rest-assured/wiki/Usage#default-parser
[481]: /rest-assured/rest-assured/wiki/Usage#default-values
[482]: /rest-assured/rest-assured/wiki/Usage#specification-re-use
[483]: /rest-assured/rest-assured/wiki/Usage#querying-requestspecification
[484]: /rest-assured/rest-assured/wiki/Usage#filters
[485]: /rest-assured/rest-assured/wiki/Usage#ordered-filters
[486]: /rest-assured/rest-assured/wiki/Usage#response-builder
[487]: /rest-assured/rest-assured/wiki/Usage#logging
[488]: /rest-assured/rest-assured/wiki/Usage#request-logging
[489]: /rest-assured/rest-assured/wiki/Usage#response-logging
[490]: /rest-assured/rest-assured/wiki/Usage#log-if-validation-fails
[491]: /rest-assured/rest-assured/wiki/Usage#blacklist-headers-from-logging
[492]: /rest-assured/rest-assured/wiki/Usage#root-path
[493]: /rest-assured/rest-assured/wiki/Usage#path-arguments
[494]: /rest-assured/rest-assured/wiki/Usage#session-support
[495]: /rest-assured/rest-assured/wiki/Usage#session-filter
[496]: /rest-assured/rest-assured/wiki/Usage#ssl
[497]: /rest-assured/rest-assured/wiki/Usage#ssl-invalid-hostname
[498]: /rest-assured/rest-assured/wiki/Usage#url-encoding
[499]: /rest-assured/rest-assured/wiki/Usage#proxy-configuration
[500]: /rest-assured/rest-assured/wiki/Usage#static-proxy-configuration
[501]: /rest-assured/rest-assured/wiki/Usage#request-specification-proxy-configuration
[502]: /rest-assured/rest-assured/wiki/Usage#detailed-configuration
[503]: /rest-assured/rest-assured/wiki/Usage#encoder-config
[504]: /rest-assured/rest-assured/wiki/Usage#avoid-adding-the-charset-to-content-type-header-automat
ically
[505]: /rest-assured/rest-assured/wiki/Usage#decoder-config
[506]: /rest-assured/rest-assured/wiki/Usage#session-config
[507]: /rest-assured/rest-assured/wiki/Usage#redirect-dsl
[508]: /rest-assured/rest-assured/wiki/Usage#connection-config
[509]: /rest-assured/rest-assured/wiki/Usage#json-config
[510]: /rest-assured/rest-assured/wiki/Usage#http-client-config
[511]: /rest-assured/rest-assured/wiki/Usage#ssl-config
[512]: /rest-assured/rest-assured/wiki/Usage#param-config
[513]: /rest-assured/rest-assured/wiki/Usage#failure-config
[514]: /rest-assured/rest-assured/wiki/Usage#csrf-config
[515]: /rest-assured/rest-assured/wiki/Usage#more-info
[516]: /rest-assured/rest-assured/wiki/Usage_Legacy
[517]: 
[518]: GettingStarted
[519]: Downloads
[520]: Usage
[521]: Usage_Legacy
[522]: snapshot
[523]: ReleaseNotes
[524]: FAQ
[525]: http://groups.google.com/group/rest-assured
