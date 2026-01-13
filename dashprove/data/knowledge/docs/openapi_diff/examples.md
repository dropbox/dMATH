# OpenAPI-diff

Compare two OpenAPI specifications (3.x) and render the difference to HTML plain text, Markdown
files, or JSON files.

[[Build]][1] [[Quality Gate Status]][2]

[[Maven Central]][3]

[[Contribute with Gitpod]][4] [[Join the Slack chat room]][5]

[[Docker Automated build]][6] [[Docker Image Version]][7]

# Requirements

* Java 8

# Feature

* Supports OpenAPI spec v3.0.
* In-depth comparison of parameters, responses, endpoint, http method (GET,POST,PUT,DELETE...)
* Supports swagger api Authorization
* Render difference of property with Expression Language
* HTML, Markdown, Asciidoc & JSON render

# Maven

Available on [Maven Central][8]

<dependency>
  <groupId>org.openapitools.openapidiff</groupId>
  <artifactId>openapi-diff-core</artifactId>
  <version>${openapi-diff-version}</version>
</dependency>

# Homebrew

Available for Mac users on [brew][9]

brew install openapi-diff

Usage instructions in [Usage -> Command line][10]

# Docker

Available on [Docker Hub][11] as `openapitools/openapi-diff`.

# docker run openapitools/openapi-diff:latest
usage: openapi-diff <old> <new>
    --asciidoc <file>           export diff as asciidoc in given file
    --debug                     Print debugging information
    --error                     Print error information
    --fail-on-changed           Fail if API changed but is backward
                                compatible
    --fail-on-incompatible      Fail only if API changes broke backward
                                compatibility
    --config-file               Config file to override default behavior. Supported file formats: .y
aml
    --config-prop               Config property to override default behavior with key:value format (
e.g. my.prop:true)
 -h,--help                      print this message
    --header <property=value>   use given header for authorisation
    --html <file>               export diff as html in given file
    --info                      Print additional information
    --json <file>               export diff as json in given file
 -l,--log <level>               use given level for log (TRACE, DEBUG,
                                INFO, WARN, ERROR, OFF). Default: ERROR
    --markdown <file>           export diff as markdown in given file
    --off                       No information printed
    --query <property=value>    use query param for authorisation
    --state                     Only output diff state: no_changes,
                                incompatible, compatible
    --text <file>               export diff as text in given file
    --trace                     be extra verbose
    --version                   print the version information and exit
    --warn                      Print warning information

## Build the image

This is only required if you want to try new changes in the Dockerfile of this project.

docker build -t local-openapi-diff .

You can replace the local image name `local-openapi-diff` by any name of your choice.

## Run an instance

In this example the `$(pwd)/core/src/test/resources` directory is mounted in the `/specs` directory
of the container in readonly mode (`ro`).

docker run --rm -t \
  -v $(pwd)/core/src/test/resources:/specs:ro \
  openapitools/openapi-diff:latest /specs/path_1.yaml /specs/path_2.yaml

The remote name `openapitools/openapi-diff` can be replaced with `local-openapi-diff` or the name
you gave to your local image.

# Usage

openapi-diff can read OpenAPI specs from JSON files or HTTP URLs.

## Command Line

$ openapi-diff --help
usage: openapi-diff <old> <new>
    --asciidoc <file>           export diff as asciidoc in given file
    --debug                     Print debugging information
    --error                     Print error information
 -h,--help                      print this message
    --header <property=value>   use given header for authorisation
    --html <file>               export diff as html in given file
    --info                      Print additional information
    --json <file>               export diff as json in given file
 -l,--log <level>               use given level for log (TRACE, DEBUG,
                                INFO, WARN, ERROR, OFF). Default: ERROR
    --markdown <file>           export diff as markdown in given file
    --off                       No information printed
    --query <property=value>    use query param for authorisation
    --state                     Only output diff state: no_changes,
                                incompatible, compatible
    --fail-on-incompatible      Fail only if API changes broke backward compatibility
    --fail-on-changed           Fail if API changed but is backward compatible
    --config-file               Config file to override default behavior. Supported file formats: .y
aml
    --config-prop               Config property to override default behavior with key:value format (
e.g. my.prop:true)
    --trace                     be extra verbose
    --version                   print the version information and exit
    --warn                      Print warning information

## Maven Plugin

Add openapi-diff to your POM to show diffs when you test your Maven project. You may opt to throw an
error if you have broken backwards compatibility or if your API has changed.

<plugin>
  <groupId>org.openapitools.openapidiff</groupId>
  <artifactId>openapi-diff-maven</artifactId>
  <version>${openapi-diff-version}</version>
  <executions>
    <execution>
      <goals>
        <goal>diff</goal>
      </goals>
      <configuration>
        <!-- Reference specification (perhaps your prod schema) -->
        <oldSpec>https://petstore3.swagger.io/api/v3/openapi.json</oldSpec>
        <!-- Specification generated by your project in the compile phase -->
        <newSpec>${project.basedir}/target/openapi.yaml</newSpec>
        <!-- Fail only if API changes broke backward compatibility (default: false) -->
        <failOnIncompatible>true</failOnIncompatible>
        <!-- Fail if API changed (default: false) -->
        <failOnChanged>true</failOnChanged>
        <!-- Supply file path for console output to file if desired. -->
        <consoleOutputFileName>${project.basedir}/../maven/target/diff.txt</consoleOutputFileName>
        <!-- Supply file path for json output to file if desired. -->
        <jsonOutputFileName>${project.basedir}/../maven/target/diff.json</jsonOutputFileName>
        <!-- Supply file path for markdown output to file if desired. -->
        <markdownOutputFileName>${project.basedir}/../maven/target/diff.md</markdownOutputFileName>
        <!-- Supply config file(s), e.g. to disable incompatibility checks. Later files override ear
lier files -->
        <configFiles>
          <configFile>my/config-file.yaml</configFile>
        </configFiles>
        <!-- Supply config properties, e.g. to disable incompatibility checks. Overrides configFiles
. -->
        <configProps>
          <incompatible.response.enum.increased>false</incompatible.response.enum.increased>
        </configProps>
      </configuration>
    </execution>
  </executions>
</plugin>

## Direct Invocation

public class Main {
    public static final String OPENAPI_DOC1 = "petstore_v3_1.json";
    public static final String OPENAPI_DOC2 = "petstore_v2_2.yaml";
        
    public static void main(String[] args) {
        ChangedOpenApi diff = OpenApiCompare.fromLocations(OPENAPI_DOC1, OPENAPI_DOC2);

        //...
    }
}

### Path Matching while comparing two OpenAPI paths

Path matching controls how paths from the old and new specs are paired during comparison
(PathsDiff.java). The default matcher (DefaultPathMatcher) obfuscates path parameter names, meaning
`/users/{id}` matches `/users/{userId}`. The default matcher fails on ambiguous signatures if the
spec contains a few paths semantically identical. In case this behaviour is not fitting your use
case, you can implement your own matching strategy.

You can plug in a custom matcher via `OpenApiDiffOptions` implementing the `PathMatcher` interface:

OpenApiDiffOptions options = OpenApiDiffOptions
    .builder()
    .pathMatcher(new MyCustomPathMatcher())
    .build();

ChangedOpenApi diff = OpenApiCompare.fromLocations(oldSpec, newSpec, null, options);

### Render difference

#### HTML

HtmlRender htmlRender = new HtmlRender("Changelog", "http://deepoove.com/swagger-diff/stylesheets/de
mo.css");
FileOutputStream outputStream = new FileOutputStream("testDiff.html");
OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
htmlRender.render(diff, outputStreamWriter);

#### Markdown

MarkdownRender markdownRender = new MarkdownRender();
FileOutputStream outputStream = new FileOutputStream("testDiff.md");
OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
markdownRender.render(diff, outputStreamWriter);

#### Asciidoc

AsciidocRender asciidocRender = new AsciidocRender();
FileOutputStream outputStream = new FileOutputStream("testDiff.adoc");
OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
asciidocRender.render(diff, outputStreamWriter);

#### JSON

JsonRender jsonRender = new JsonRender();
FileOutputStream outputStream = new FileOutputStream("testDiff.json");
OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
jsonRender.render(diff, outputStreamWriter);

### Extensions

This project uses Java Service Provider Interface (SPI) so additional extensions can be added.

To build your own extension, you simply need to create a
`src/main/resources/META-INF/services/org.openapitools.openapidiff.core.compare.ExtensionDiff` file
with the full classname of your implementation. Your class must also implement the
`org.openapitools.openapidiff.core.compare.ExtensionDiff` interface. Then, including your library
with the `openapi-diff` module will cause it to be triggered automatically.

# Examples

### CLI Output

`==========================================================================
==                            API CHANGE LOG                            ==
==========================================================================
                             Swagger Petstore                             
--------------------------------------------------------------------------
--                              What's New                              --
--------------------------------------------------------------------------
- GET    /pet/{petId}

--------------------------------------------------------------------------
--                            What's Deleted                            --
--------------------------------------------------------------------------
- POST   /pet/{petId}

--------------------------------------------------------------------------
--                          What's Deprecated                           --
--------------------------------------------------------------------------
- GET    /user/logout

--------------------------------------------------------------------------
--                            What's Changed                            --
--------------------------------------------------------------------------
- PUT    /pet
  Request:
        - Deleted application/xml
        - Changed application/json
          Schema: Backward compatible
- POST   /pet
  Parameter:
    - Add tags in query
  Request:
        - Changed application/xml
          Schema: Backward compatible
        - Changed application/json
          Schema: Backward compatible
- GET    /pet/findByStatus
  Parameter:
    - Deprecated status in query
  Return Type:
    - Changed 200 OK
      Media types:
        - Changed application/xml
          Schema: Broken compatibility
        - Changed application/json
          Schema: Broken compatibility
- GET    /pet/findByTags
  Return Type:
    - Changed 200 OK
      Media types:
        - Changed application/xml
          Schema: Broken compatibility
        - Changed application/json
          Schema: Broken compatibility
- DELETE /pet/{petId}
  Parameter:
    - Add newHeaderParam in header
- POST   /pet/{petId}/uploadImage
  Parameter:
    - Changed petId in path
- POST   /user
  Request:
        - Changed application/json
          Schema: Backward compatible
- POST   /user/createWithArray
  Request:
        - Changed application/json
          Schema: Backward compatible
- POST   /user/createWithList
  Request:
        - Changed application/json
          Schema: Backward compatible
- GET    /user/login
  Parameter:
    - Delete password in query
- GET    /user/logout
- GET    /user/{username}
  Return Type:
    - Changed 200 OK
      Media types:
        - Changed application/xml
          Schema: Broken compatibility
        - Changed application/json
          Schema: Broken compatibility
- PUT    /user/{username}
  Request:
        - Changed application/json
          Schema: Backward compatible
--------------------------------------------------------------------------
--                                Result                                --
--------------------------------------------------------------------------
                 API changes broke backward compatibility                 
--------------------------------------------------------------------------
`

### Markdown

### What's New
---
* `GET` /pet/{petId} Find pet by ID

### What's Deleted
---
* `POST` /pet/{petId} Updates a pet in the store with form data

### What's Deprecated
---
* `GET` /user/logout Logs out current logged in user session

### What's Changed
---
* `PUT` /pet Update an existing pet  
    Request

        Deleted request body : [application/xml]
        Changed response : [application/json]
* `POST` /pet Add a new pet to the store  
    Parameter

        Add tags //add new query param demo
    Request

        Changed response : [application/xml]
        Changed response : [application/json]
* `GET` /pet/findByStatus Finds Pets by status  
    Parameter

    Return Type

        Changed response : [200] //successful operation
* `GET` /pet/findByTags Finds Pets by tags  
    Return Type

        Changed response : [200] //successful operation
* `DELETE` /pet/{petId} Deletes a pet  
    Parameter

        Add newHeaderParam
* `POST` /pet/{petId}/uploadImage uploads an image for pet  
    Parameter

        petId Notes ID of pet to update change into ID of pet to update, default false
* `POST` /user Create user  
    Request

        Changed response : [application/json]
* `POST` /user/createWithArray Creates list of users with given input array  
    Request

        Changed response : [application/json]
* `POST` /user/createWithList Creates list of users with given input array  
    Request

        Changed response : [application/json]
* `GET` /user/login Logs user into the system  
    Parameter

        Delete password //The password for login in clear text
* `GET` /user/logout Logs out current logged in user session  
* `PUT` /user/{username} Updated user  
    Request

        Changed response : [application/json]
* `GET` /user/{username} Get user by user name  
    Return Type

        Changed response : [200] //successful operation

### JSON

{
    "changedElements": [...],
    "changedExtensions": null,
    "changedOperations": [...],
    "compatible": false,
    "deprecatedEndpoints": [...],
    "different": true,
    "incompatible": true,
    "missingEndpoints": [...],
    "newEndpoints": [
        {
            "method": "GET",
            "operation": {
                "callbacks": null,
                "deprecated": null,
                "description": "Returns a single pet",
                "extensions": null,
                "externalDocs": null,
                "operationId": "getPetById",
                "parameters": [
                    {
                        "$ref": null,
                        "allowEmptyValue": null,
                        "allowReserved": null,
                        "content": null,
                        "deprecated": null,
                        "description": "ID of pet to return",
                        "example": null,
                        "examples": null,
                        "explode": false,
                        "extensions": null,
                        "in": "path",
                        "name": "petId",
                        "required": true,
                        "schema": {...},
                        "style": "SIMPLE"
                    }
                ],
                "requestBody": null,
                "responses": {...},
                "security": [
                    {
                        "api_key": []
                    }
                ],
                "servers": null,
                "summary": "Find pet by ID",
                "tags": [
                    "pet"
                ]
            },
            "path": null,
            "pathUrl": "/pet/{petId}",
            "summary": "Find pet by ID"
        }
    ],
    "newSpecOpenApi": {...},
    "oldSpecOpenApi": {...},
    "unchanged": false
}

# License

openapi-diff is released under the Apache License 2.0.

# Thanks

* Adarsh Sharma / [adarshsharma][12]
* Quentin Desramé / [quen2404][13]
* [Sayi][14] for his project [swagger-diff][15] which was a source of inspiration for this tool

[1]: https://github.com/OpenAPITools/openapi-diff/actions?query=branch%3Amaster+workflow%3A%22Main+B
uild%22
[2]: https://sonarcloud.io/dashboard?id=OpenAPITools_openapi-diff
[3]: https://search.maven.org/artifact/org.openapitools.openapidiff/openapi-diff-core
[4]: https://gitpod.io/#https://github.com/OpenAPITools/openapi-diff
[5]: https://join.slack.com/t/openapi-generator/shared_invite/zt-12jxxd7p2-XUeQM~4pzsU9x~eGLQqX2g
[6]: https://hub.docker.com/r/openapitools/openapi-diff
[7]: https://hub.docker.com/r/openapitools/openapi-diff/tags
[8]: https://search.maven.org/artifact/org.openapitools.openapidiff/openapi-diff-core
[9]: https://formulae.brew.sh/formula/openapi-diff
[10]: #command-line
[11]: https://hub.docker.com/r/openapitools/openapi-diff/
[12]: https://github.com/adarshsharma
[13]: https://github.com/quen2404
[14]: https://github.com/Sayi
[15]: https://github.com/Sayi/swagger-diff
