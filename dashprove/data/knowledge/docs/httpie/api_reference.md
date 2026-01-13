HTTPie (pronounced *aitch-tee-tee-pie*) is a command-line HTTP client. Its goal is to make CLI
interaction with web services as human-friendly as possible. HTTPie is designed for testing,
debugging, and generally interacting with APIs & HTTP servers. The `http` & `https` commands allow
for creating and sending arbitrary HTTP requests. They use simple and natural syntax and provide
formatted and colorized output.

## [Main features][1]

* Expressive and intuitive syntax
* Formatted and colorized terminal output
* Built-in JSON support
* Forms and file uploads
* HTTPS, proxies, and authentication
* Arbitrary request data
* Custom headers
* Persistent sessions
* Wget-like downloads
* Linux, macOS, Windows, and FreeBSD support
* Plugins
* Documentation
* Test coverage

## [Installation][2]

* [Universal][3]
* [macOS][4]
* [Windows][5]
* [Linux][6]
* [FreeBSD][7]

### [Universal][8]

#### [PyPI][9]

Please make sure you have Python 3.7 or newer (`python --version`).

`# Install httpie
$ python -m pip install --upgrade pip wheel
$ python -m pip install httpie`
`# Upgrade httpie
$ python -m pip install --upgrade pip wheel
$ python -m pip install --upgrade httpie`

### [macOS][10]

#### [Homebrew][11]

To install [Homebrew][12], see [its installation][13].

`# Install httpie
$ brew update
$ brew install httpie`
`# Upgrade httpie
$ brew update
$ brew upgrade httpie`

#### [MacPorts][14]

To install [MacPorts][15], see [its installation][16].

`# Install httpie
$ port selfupdate
$ port install httpie`
`# Upgrade httpie
$ port selfupdate
$ port upgrade httpie`

### [Windows][17]

#### [Chocolatey][18]

To install [Chocolatey][19], see [its installation][20].

`# Install httpie
$ choco install httpie`
Run
`# Upgrade httpie
$ choco upgrade httpie`
Run

### [Linux][21]

#### [Debian and Ubuntu][22]

Also works for other Debian-derived distributions like MX Linux, Linux Mint, deepin, Pop!_OS, KDE
neon, Zorin OS, elementary OS, Kubuntu, Devuan, Linux Lite, Peppermint OS, Lubuntu, antiX, Xubuntu,
etc.

`# Install httpie
$ curl -SsL https://packages.httpie.io/deb/KEY.gpg | sudo gpg --dearmor -o /usr/share/keyrings/httpi
e.gpg
$ echo "deb [arch=amd64 signed-by=/usr/share/keyrings/httpie.gpg] https://packages.httpie.io/deb ./"
 | sudo tee /etc/apt/sources.list.d/httpie.list > /dev/null
$ sudo apt update
$ sudo apt install httpie`
`# Upgrade httpie
$ sudo apt update && sudo apt upgrade httpie`
Run

#### [Fedora][23]

`# Install httpie
$ dnf install httpie`
Run
`# Upgrade httpie
$ dnf upgrade httpie`
Run

#### [CentOS and RHEL][24]

Also works for other RHEL-derived distributions like ClearOS, Oracle Linux, etc.

`# Install httpie
$ yum install epel-release
$ yum install httpie`
`# Upgrade httpie
$ yum upgrade httpie`
Run

#### [Single binary executables][25]

Get the standalone HTTPie Linux executables when you don't want to go through the full installation
process.

`# Install httpie
$ https --download packages.httpie.io/binaries/linux/http-latest -o http
$ ln -ls ./http ./https
$ chmod +x ./http ./https`
`# Upgrade httpie
$ https --download packages.httpie.io/binaries/linux/http-latest -o http`
Run

#### [Snapcraft (Linux)][26]

To install [Snapcraft][27], see [its installation][28].

`# Install httpie
$ snap install httpie`
Run
`# Upgrade httpie
$ snap refresh httpie`
Run

#### [Linuxbrew][29]

To install [Linuxbrew][30], see [its installation][31].

`# Install httpie
$ brew update
$ brew install httpie`
`# Upgrade httpie
$ brew update
$ brew upgrade httpie`

#### [Arch Linux][32]

Also works for other Arch-derived distributions like ArcoLinux, EndeavourOS, Artix Linux, etc.

`# Install httpie
$ pacman -Syu httpie`
Run
`# Upgrade httpie
$ pacman -Syu`

### [FreeBSD][33]

#### [FreshPorts][34]

`# Install httpie
$ pkg install www/py-httpie`
Run
`# Upgrade httpie
$ pkg upgrade www/py-httpie`
Run

### [Unstable version][35]

If you want to try out the latest version of HTTPie that hasn't been officially released yet, you
can install the development or unstable version directly from the master branch on GitHub. However,
keep in mind that the development version is a work in progress and may not be as reliable as the
stable version.

You can use the following command to install the development version of HTTPie on Linux, macOS,
Windows, or FreeBSD operating systems. With this command, the code present in the `master` branch is
downloaded and installed using `pip`.

`$ python -m pip install --upgrade https://github.com/httpie/cli/archive/master.tar.gz`
Run

There are other ways to install the development version of HTTPie on macOS and Linux.

You can install it using Homebrew by running the following commands:

`$ brew uninstall --force httpie
$ brew install --HEAD httpie`

You can install it using Snapcraft by running the following commands:

`$ snap remove httpie
$ snap install httpie --edge`

To verify the installation, you can compare the [version identifier on GitHub][36] with the one
available on your machine. You can check the version of HTTPie on your machine by using the command
`http --version`.

`$ http --version
# 3.X.X.dev0`
Run

Note that on your machine, the version name will have the `.dev0` suffix.

## [Usage][37]

Hello World:

`$ https httpie.io/hello`
Run

Synopsis:

`$ http [flags] [METHOD] URL [ITEM [ITEM]]`

See also `http --help` (and for systems where man pages are available, you can use `man http`).

### [Examples][38]

Custom [HTTP method][39], [HTTP headers][40] and [JSON][41] data:

`$ http PUT pie.dev/put X-API-Token:123 name=John`
Run

Submitting [forms][42]:

`$ http -f POST pie.dev/post hello=World`
Run

See the request that is being sent using one of the [output options][43]:

`$ http -v pie.dev/get`
Run

Build and print a request without sending it using [offline mode][44]:

`$ http --offline pie.dev/post hello=offline`
Run

Use [GitHub API][45] to post a comment on an [issue][46] with [authentication][47]:

`$ http -a USERNAME POST https://api.github.com/repos/httpie/cli/issues/83/comments body='HTTPie is 
awesome! :heart:'`
Run

Upload a file using [redirected input][48]:

`$ http pie.dev/post < files/data.json`
Run

Download a file and save it via [redirected output][49]:

`$ http pie.dev/image/png > image.png`
Run

Download a file `wget` style:

`$ http --download pie.dev/image/png`
Run

Use named [sessions][50] to make certain aspects of the communication persistent between requests to
the same host:

`$ http --session=logged-in -a username:password pie.dev/get API-Key:123`
Run
`$ http --session=logged-in pie.dev/headers`
Run

Set a custom `Host` header to work around missing DNS records:

`$ http localhost:8000 Host:example.com`
Run

## [HTTP method][51]

The name of the HTTP method comes right before the URL argument:

`$ http DELETE pie.dev/delete`
Run

Which looks similar to the actual `Request-Line` that is sent:

`DELETE /delete HTTP/1.1`

In addition to the standard methods (`GET`, `POST`, `HEAD`, `PUT`, `PATCH`, `DELETE`, etc.), you can
use custom method names, for example:

`$ http AHOY pie.dev/post`
Run

There are no restrictions regarding which request methods can include a body. You can send an empty
`POST` request:

`$ http POST pie.dev/post`
Run

You can also make `GET` requests containing a body:

`$ http GET pie.dev/get hello=world`
Run

### [Optional `GET` and `POST`][52]

The `METHOD` argument is optional, and when you don’t specify it, HTTPie defaults to:

* `GET` for requests without body
* `POST` for requests with body

Here we don’t specify any request data, so both commands will send the same `GET` request:

`$ http GET pie.dev/get`
Run
`$ http pie.dev/get`
Run

Here, on the other hand, we do have some data, so both commands will make the same `POST` request:

`$ http POST pie.dev/post hello=world`
Run
`$ http pie.dev/post hello=world`
Run

## [Request URL][53]

The only information HTTPie needs to perform a request is a URL.

The default scheme is `http://` and can be omitted from the argument:

`$ http example.org
# → http://example.org`
Run

HTTPie also installs an `https` executable, where the default scheme is `https://`:

`$ https example.org
# → https://example.org`
Run

When you paste a URL into the terminal, you can even keep the `://` bit in the URL argument to
quickly convert the URL into an HTTPie call just by adding a space after the protocol name.

`$ https ://example.org
# → https://example.org`
Run
`$ http ://example.org
# → http://example.org`
Run

### [Querystring parameters][54]

If you find yourself manually constructing URLs with querystring parameters on the terminal, you may
appreciate the `param==value` syntax for appending URL parameters.

With that, you don’t have to worry about escaping the `&` separators for your shell. Additionally,
any special characters in the parameter name or value get automatically URL-escaped (as opposed to
the parameters specified in the full URL, which HTTPie doesn’t modify).

`$ http https://api.github.com/search/repositories q==httpie per_page==1`
Run
`GET /search/repositories?q=httpie&per_page=1 HTTP/1.1`

You can even retrieve the `value` from a file by using the `param==@file` syntax. This would also
effectively strip the newlines from the end. See [file based separators][55] for more examples.

`$ http pie.dev/get text==@files/text.txt`
Run

### [URL shortcuts for `localhost`][56]

Additionally, curl-like shorthand for localhost is supported. This means that, for example, `:3000`
would expand to `http://localhost:3000`. If the port is omitted, then port 80 is assumed.

`$ http :/foo`
Run
`GET /foo HTTP/1.1
Host: localhost`
`$ http :3000/bar`
Run
`GET /bar HTTP/1.1
Host: localhost:3000`
`$ http :`
Run
`GET / HTTP/1.1
Host: localhost`

### [Other default schemes][57]

When HTTPie is invoked as `https` then the default scheme is `https://` (`$ https example.org` will
make a request to `https://example.org`).

You can also use the `--default-scheme <URL_SCHEME>` option to create shortcuts for other protocols
than HTTP (possibly supported via [plugins][58]). Example for the [httpie-unixsocket][59] plugin:

`# Before
$ http http+unix://%2Fvar%2Frun%2Fdocker.sock/info`
Run
`# Create an alias
$ alias http-unix='http --default-scheme="http+unix"'`
Run
`# Now the scheme can be omitted
$ http-unix %2Fvar%2Frun%2Fdocker.sock/info`
Run

### [`--path-as-is`][60]

The standard behavior of HTTP clients is to normalize the path portion of URLs by squashing dot
segments as a typical filesystem would:

`$ http -v example.org/./../../etc/password`
Run
`GET /etc/password HTTP/1.1`

The `--path-as-is` option allows you to disable this behavior:

`$ http --path-as-is -v example.org/./../../etc/password`
Run
`GET /../../etc/password HTTP/1.1`

## [Request items][61]

There are a few different *request item* types that provide a convenient mechanism for specifying
HTTP headers, JSON and form data, files, and URL parameters. This is a very practical way of
constructing HTTP requests from scratch on the CLI.

Each *request item* is simply a key/value pair separated with the following characters: `:`
(headers), `=` (data field, e.g., JSON, form), `:=` (raw data field) `==` (query parameters), `@`
(file upload).

`$ http PUT pie.dev/put \
    X-Date:today \                     # Header
    token==secret \                    # Query parameter
    name=John \                        # Data field
    age:=29                            # Raw JSON`
Run

────────────────────┬───────────────────────────────────────────────────────────────────────────────
Item Type           │Description                                                                    
────────────────────┼───────────────────────────────────────────────────────────────────────────────
HTTP Headers        │Arbitrary HTTP header, e.g. `X-API-Token:123`                                  
`Name:Value`        │                                                                               
────────────────────┼───────────────────────────────────────────────────────────────────────────────
URL parameters      │Appends the given name/value pair as a querystring parameter to the URL. The   
`name==value`       │`==` separator is used.                                                        
────────────────────┼───────────────────────────────────────────────────────────────────────────────
Data Fields         │Request data fields to be serialized as a JSON object (default), to be         
`field=value`       │form-encoded (with `--form, -f`), or to be serialized as `multipart/form-data` 
                    │(with `--multipart`)                                                           
────────────────────┼───────────────────────────────────────────────────────────────────────────────
Raw JSON fields     │Useful when sending JSON and one or more fields need to be a `Boolean`,        
`field:=json`       │`Number`, nested `Object`, or an `Array`, e.g., `meals:='["ham","spam"]'` or   
                    │`pies:='[1,2,3]'` (note the quotes)                                            
────────────────────┼───────────────────────────────────────────────────────────────────────────────
File upload fields  │Only available with `--form`, `-f` and `--multipart`. For example              
`field@/dir/file`,  │`screenshot@~/Pictures/img.png`, or `'[[email                                  
`field@file;type=mim│protected]][62];type=text/markdown'`. With `--form`, the presence of a file    
e`                  │field results in a `--multipart` request                                       
────────────────────┴───────────────────────────────────────────────────────────────────────────────

Note that the structured data fields aren’t the only way to specify request data: [raw request
body][63] is a mechanism for passing arbitrary request data.

### [File based separators][64]

Using file contents as values for specific fields is a very common use case, which can be achieved
through adding the `@` suffix to the operators above. For example, instead of using a static string
as the value for some header, you can use `:@` operator to pass the desired value from a file.

`$ http POST pie.dev/post \
    X-Data:@files/text.txt             # Read a header from a file
    token==@files/text.txt             # Read a query parameter from a file
    name=@files/text.txt               # Read a data field’s value from a file
    bookmarks:=@files/data.json        # Embed a JSON object from a file`

### [Escaping rules][65]

You can use `\` to escape characters that shouldn’t be used as separators (or parts thereof). For
instance, `foo\==bar` will become a data key/value pair (`foo=` and `bar`) instead of a URL
parameter.

Often it is necessary to quote the values, e.g. `foo='bar baz'`.

If any of the field names or headers starts with a minus (e.g. `-fieldname`), you need to place all
such items after the special token `--` to prevent confusion with `--arguments`:

`$ http pie.dev/post -- -name-starting-with-dash=foo -Unusual-Header:bar`
Run
`POST /post HTTP/1.1
-Unusual-Header: bar
Content-Type: application/json

{
    "-name-starting-with-dash": "foo"
}`

## [JSON][66]

JSON is the *lingua franca* of modern web services, and it is also the **implicit content type**
HTTPie uses by default.

Simple example:

`$ http PUT pie.dev/put name=John [[email protected]][67]`
Run
`PUT / HTTP/1.1
Accept: application/json, */*;q=0.5
Accept-Encoding: gzip, deflate
Content-Type: application/json
Host: pie.dev

{
    "name": "John",
    "email": "[[email protected]][68]"
}`

### [Default behavior][69]

If your command includes some data [request items][70], they are serialized as a JSON object by
default. HTTPie also automatically sets the following headers, both of which can be overwritten:

────────────┬───────────────────────────
Header      │Value                      
────────────┼───────────────────────────
`Content-Typ│`application/json`         
e`          │                           
────────────┼───────────────────────────
`Accept`    │`application/json,         
            │*/*;q=0.5`                 
────────────┴───────────────────────────

### [Explicit JSON][71]

You can use `--json, -j` to explicitly set `Accept` to `application/json` regardless of whether you
are sending data (it’s a shortcut for setting the header via the usual header notation: `http url
Accept:'application/json, */*;q=0.5'`). Additionally, HTTPie will try to detect JSON responses even
when the `Content-Type` is incorrectly `text/plain` or unknown.

### [Non-string JSON fields][72]

Non-string JSON fields use the `:=` separator, which allows you to embed arbitrary JSON data into
the resulting JSON object. Additionally, text and raw JSON files can also be embedded into fields
using `=@` and `:=@`:

`$ http PUT pie.dev/put \
    name=John \                        # String (default)
    age:=29 \                          # Raw JSON — Number
    married:=false \                   # Raw JSON — Boolean
    hobbies:='["http", "pies"]' \      # Raw JSON — Array
    favorite:='{"tool": "HTTPie"}' \   # Raw JSON — Object
    bookmarks:=@files/data.json \      # Embed JSON file
    description=@files/text.txt        # Embed text file`
Run
`PUT /person/1 HTTP/1.1
Accept: application/json, */*;q=0.5
Content-Type: application/json
Host: pie.dev

{
    "age": 29,
    "hobbies": [
        "http",
        "pies"
    ],
    "description": "John is a nice guy who likes pies.",
    "married": false,
    "name": "John",
    "favorite": {
        "tool": "HTTPie"
    },
    "bookmarks": {
        "HTTPie": "https://httpie.org",
    }
}`

The `:=`/`:=@` syntax is JSON-specific. You can switch your request to `--form` or `--multipart`,
and string, float, and number values will continue to be serialized (as string form values). Other
JSON types, however, are not allowed with `--form` or `--multipart`.

### [Nested JSON][73]

If your use case involves sending complex JSON objects as part of the request body, HTTPie can help
you build them right from your terminal. You still use the existing data field operators (`=`/`:=`)
but instead of specifying a top-level field name (like `key=value`), you specify a path declaration.
This tells HTTPie where and how to put the given value inside an object:

`http pie.dev/post \
  platform[name]=HTTPie \
  platform[about][mission]='Make APIs simple and intuitive' \
  platform[about][homepage]=httpie.io \
  platform[about][homepage]=httpie.io \
  platform[about][stars]:=54000 \
  platform[apps][]=Terminal \
  platform[apps][]=Desktop \
  platform[apps][]=Web \
  platform[apps][]=Mobile`
Run
`{
    "platform": {
        "name": "HTTPie",
        "about": {
            "mission": "Make APIs simple and intuitive",
            "homepage": "httpie.io",
            "stars": 54000
        },
        "apps": [
            "Terminal",
            "Desktop",
            "Web",
            "Mobile"
        ]
    }
}`

#### [Introduction][74]

Let’s start with a simple example, and build a simple search query:

`$ http --offline --print=B pie.dev/post \
  category=tools \
  search[type]=id \
  search[id]:=1`
Run

In the example above, the `search[type]` is an instruction for creating an object called `search`,
and setting the `type` field of it to the given value (`"id"`).

Also note that, just as the regular syntax, you can use the `:=` operator to directly pass raw JSON
values (e.g., numbers in the case above).

`{
    "category": "tools",
    "search": {
        "id": 1,
        "type": "id"
    }
}`

Building arrays is also possible, through `[]` suffix (an append operation). This tells HTTPie to
create an array in the given path (if there is not one already), and append the given value to that
array.

`$ http --offline --print=B pie.dev/post \
  category=tools \
  search[type]=keyword \
  search[keywords][]=APIs \
  search[keywords][]=CLI`
Run
`{
    "category": "tools",
    "search": {
        "keywords": [
            "APIs",
            "CLI"
        ],
        "type": "keyword"
    }
}`

If you want to explicitly specify the position of elements inside an array, you can simply pass the
desired index as the path:

`$ http --offline --print=B pie.dev/post \
  category=tools \
  search[type]=keyword \
  search[keywords][1]=APIs \
  search[keywords][0]=CLI`
Run
`{
    "category": "tools",
    "search": {
        "keywords": [
            "CLIs",
            "API"
        ],
        "type": "keyword"
    }
}`

If there are any missing indexes, HTTPie will nullify them in order to create a concrete object that
can be sent:

`$ http --offline --print=B pie.dev/post \
  category=tools \
  search[type]=platforms \
  search[platforms][]=Terminal \
  search[platforms][1]=Desktop \
  search[platforms][3]=Mobile`
Run
`{
    "category": "tools",
    "search": {
        "platforms": [
            "Terminal",
            "Desktop",
            null,
            "Mobile"
        ],
        "type": "platforms"
    }
}`

It is also possible to embed raw JSON to a nested structure, for example:

`$ http --offline --print=B pie.dev/post \
  category=tools \
  search[type]=platforms \
  'search[platforms]:=["Terminal", "Desktop"]' \
  search[platforms][]=Web \
  search[platforms][]=Mobile`
Run
`{
    "category": "tools",
    "search": {
        "platforms": [
            "Terminal",
            "Desktop",
            "Web",
            "Mobile"
        ],
        "type": "platforms"
    }
}`

And just to demonstrate all of these features together, let’s create a very deeply nested JSON
object:

`$ http PUT pie.dev/put \
    shallow=value \                                # Shallow key-value pair
    object[key]=value \                            # Nested key-value pair
    array[]:=1 \                                   # Array — first item
    array[1]:=2 \                                  # Array — second item
    array[2]:=3 \                                  # Array — append (third item)
    very[nested][json][3][httpie][power][]=Amaze   # Nested object`
Run

#### [Advanced usage][75]

[Top level arrays][76]

If you want to send an array instead of a regular object, you can simply do that by omitting the
starting key:

`$ http --offline --print=B pie.dev/post \
    []:=1 \
    []:=2 \
    []:=3`
Run
`[
    1,
    2,
    3
]`

You can also apply the nesting to the items by referencing their index:

`http --offline --print=B pie.dev/post \
    [0][type]=platform [0][name]=terminal \
    [1][type]=platform [1][name]=desktop`
Run
`[
    {
        "type": "platform",
        "name": "terminal"
    },
    {
        "type": "platform",
        "name": "desktop"
    }
]`

Sending scalar JSON types (a single `null`, `true`, `false`, string or number) as the top-level
object is impossible using the key/value syntax. But you can still pass it via
[`--raw='<value>'`][77].

[Escaping behavior][78]

Nested JSON syntax uses the same [escaping rules][79] as the terminal. There are 3 special
characters, and 1 special token that you can escape.

If you want to send a bracket as is, escape it with a backslash (`\`):

`$ http --offline --print=B pie.dev/post \
  'foo\[bar\]:=1' \
  'baz[\[]:=2' \
  'baz[\]]:=3'`
`{
    "baz": {
        "[": 2,
        "]": 3
    },
    "foo[bar]": 1
}`

If you want to send the literal backslash character (`\`), escape it with another backslash:

`$ http --offline --print=B pie.dev/post \
  'backslash[\\]:=1'`
Run
`{
    "backslash": {
        "\\": 1
    }
}`

A regular integer in a path (e.g `[10]`) means an array index; but if you want it to be treated as a
string, you can escape the whole number by using a backslash (`\`) prefix.

`$ http --offline --print=B pie.dev/post \
  'object[\1]=stringified' \
  'object[\100]=same' \
  'array[1]=indexified'`
Run
`{
    "array": [
        null,
        "indexified"
    ],
    "object": {
        "1": "stringified",
        "100": "same"
    }
}`
[Guiding syntax errors][80]

If you make a typo or forget to close a bracket, the errors will guide you to fix it. For example:

`$ http --offline --print=B pie.dev/post \
  'foo[bar]=OK' \
  'foo[baz][quux=FAIL'`
Run
`HTTPie Syntax Error: Expecting ']'
foo[baz][quux
             ^`

You can follow to given instruction (adding a `]`) and repair your expression.

[Type safety][81]

Each container path (e.g., `x[y][z]` in `x[y][z][1]`) has a certain type, which gets defined with
the first usage and can’t be changed after that. If you try to do a key-based access to an array or
an index-based access to an object, HTTPie will error out:

`$ http --offline --print=B pie.dev/post \
  'array[]:=1' \
  'array[]:=2' \
  'array[key]:=3'
HTTPie Type Error: Can't perform 'key' based access on 'array' which has a type of 'array' but this 
operation requires a type of 'object'.
array[key]
     ^^^^^`

Type Safety does not apply to value overrides, for example:

`$ http --offline --print=B pie.dev/post \
  user[name]:=411     # Defined as an integer
  user[name]=string   # Overridden with a string`
`{
    "user": {
        "name": "string"
    }
}`

### [Raw JSON][82]

For very complex JSON structures, it may be more convenient to [pass it as raw request body][83],
for example:

`$ echo -n '{"hello": "world"}' | http POST pie.dev/post`
Run
`$ http POST pie.dev/post < files/data.json`
Run

## [Forms][84]

Submitting forms is very similar to sending [JSON][85] requests. Often the only difference is in
adding the `--form, -f` option, which ensures that data fields are serialized as key-value tuples
separated by '&', with a '=' between the key and the value. In addition `Content-Type` is set to
`application/x-www-form-urlencoded; charset=utf-8`. It is possible to make form data the implicit
content type instead of JSON via the [config][86] file.

### [Regular forms][87]

`$ http --form POST pie.dev/post name='John Smith'`
Run
`POST /post HTTP/1.1
Content-Type: application/x-www-form-urlencoded; charset=utf-8

name=John+Smith`

### [File upload forms][88]

If one or more file fields is present, the serialization and content type is `multipart/form-data`:

`$ http -f POST pie.dev/post name='John Smith' cv@~/files/data.xml`
Run

The request above is the same as if the following HTML form were submitted:

`<form enctype="multipart/form-data" method="post" action="http://example.com/jobs">
    <input type="text" name="name" />
    <input type="file" name="cv" />
</form>`

Please note that `@` is used to simulate a file upload form field, whereas `=@` just embeds the file
content as a regular text field value.

When uploading files, their content type is inferred from the file name. You can manually override
the inferred content type:

`$ http -f POST pie.dev/post name='John Smith' cv@'~/files/data.bin;type=application/pdf'`
Run

To perform a `multipart/form-data` request even without any files, use `--multipart` instead of
`--form`:

`$ http --multipart --offline example.org hello=world`
Run
`POST / HTTP/1.1
Content-Length: 129
Content-Type: multipart/form-data; boundary=c31279ab254f40aeb06df32b433cbccb
Host: example.org

--c31279ab254f40aeb06df32b433cbccb
Content-Disposition: form-data; name="hello"

world
--c31279ab254f40aeb06df32b433cbccb--`

File uploads are always streamed to avoid memory issues with large files.

By default, HTTPie uses a random unique string as the multipart boundary, but you can use
`--boundary` to specify a custom string instead:

`$ http --form --multipart --boundary=xoxo --offline example.org hello=world`
Run
`POST / HTTP/1.1
Content-Length: 129
Content-Type: multipart/form-data; boundary=xoxo
Host: example.org

--xoxo
Content-Disposition: form-data; name="hello"

world
--xoxo--`

If you specify a custom `Content-Type` header without including the boundary bit, HTTPie will add
the boundary value (explicitly specified or auto-generated) to the header automatically:

`$ http --form --multipart --offline example.org hello=world Content-Type:multipart/letter`
Run
`POST / HTTP/1.1
Content-Length: 129
Content-Type: multipart/letter; boundary=c31279ab254f40aeb06df32b433cbccb
Host: example.org

--c31279ab254f40aeb06df32b433cbccb
Content-Disposition: form-data; name="hello"

world
--c31279ab254f40aeb06df32b433cbccb--`

## [HTTP headers][89]

To set custom headers you can use the `Header:Value` notation:

`$ http pie.dev/headers User-Agent:Bacon/1.0 'Cookie:valued-visitor=yes;foo=bar' \
    X-Foo:Bar Referer:https://httpie.org/`
Run
`GET /headers HTTP/1.1
Accept: */*
Accept-Encoding: gzip, deflate
Cookie: valued-visitor=yes;foo=bar
Host: pie.dev
Referer: https://httpie.org/
User-Agent: Bacon/1.0
X-Foo: Bar`

### [Default request headers][90]

There are a couple of default headers that HTTPie sets:

`GET / HTTP/1.1
Accept: */*
Accept-Encoding: gzip, deflate
User-Agent: HTTPie/<version>
Host: <taken-from-URL>`

All of these can be overwritten or unset (see below).

### [Reading headers from a file][91]

You can read headers from a file by using the `:@` operator. This would also effectively strip the
newlines from the end. See [file based separators][92] for more examples.

`$ http pie.dev/headers X-Data:@files/text.txt`
Run

### [Empty headers and header un-setting][93]

To unset a previously specified header (such a one of the default headers), use `Header:`:

`$ http pie.dev/headers Accept: User-Agent:`
Run

To send a header with an empty value, use `Header;`, with a semicolon:

`$ http pie.dev/headers 'Header;'`
Run

Please note that some internal headers, such as `Content-Length`, can’t be unset if they are
automatically added by the client itself.

### [Multiple header values with the same name][94]

If the request is sent with multiple headers that are sharing the same name, then the HTTPie will
send them individually.

`http --offline example.org Cookie:one Cookie:two`
Run
`GET / HTTP/1.1
Cookie: one
Cookie: two`

It is also possible to pass a single header value pair, where the value is a comma separated list of
header values. Then the client will send it as a single header.

`http --offline example.org Numbers:one,two`
Run
`GET / HTTP/1.1
Numbers: one,two`

Also be aware that if the current session contains any headers they will get overwritten by
individual commands when sending a request instead of being joined together.

### [Limiting response headers][95]

The `--max-headers=n` option allows you to control the number of headers HTTPie reads before giving
up (the default `0`, i.e., there’s no limit).

`$ http --max-headers=100 pie.dev/get`
Run

## [Offline mode][96]

Use `--offline` to construct HTTP requests without sending them anywhere. With `--offline`, HTTPie
builds a request based on the specified options and arguments, prints it to `stdout`, and then
exits. It works completely offline; no network connection is ever made. This has a number of use
cases, including:

Generating API documentation examples that you can copy & paste without sending a request:

`$ http --offline POST server.chess/api/games API-Key:ZZZ w=magnus b=hikaru t=180 i=2`
Run
`$ http --offline MOVE server.chess/api/games/123 API-Key:ZZZ p=b a=R1a3 t=77`
Run

Generating raw requests that can be sent with any other client:

`# 1. save a raw request to a file:
$ http --offline POST pie.dev/post hello=world > request.http`
Run
`# 2. send it over the wire with, for example, the fantastic netcat tool:
$ nc pie.dev 80 < request.http`
Run

You can also use the `--offline` mode for debugging and exploring HTTP and HTTPie, and for “dry
runs”.

`--offline` has the side effect of automatically activating `--print=HB`, i.e., both the request
headers and the body are printed. You can customize the output with the usual [output options][97],
with the exception where there is no response to be printed. You can use `--offline` in combination
with all the other options (e.g. `--session`).

## [Cookies][98]

HTTP clients send cookies to the server as regular [HTTP headers][99]. That means, HTTPie does not
offer any special syntax for specifying cookies — the usual `Header:Value` notation is used:

Send a single cookie:

`$ http pie.dev/cookies Cookie:sessionid=foo`
Run
`GET / HTTP/1.1
Accept: */*
Accept-Encoding: gzip, deflate
Connection: keep-alive
Cookie: sessionid=foo
Host: pie.dev
User-Agent: HTTPie/0.9.9`

Send multiple cookies (note: the header is quoted to prevent the shell from interpreting the `;`):

`$ http pie.dev/cookies 'Cookie:sessionid=foo;another-cookie=bar'`
Run
`GET / HTTP/1.1
Accept: */*
Accept-Encoding: gzip, deflate
Connection: keep-alive
Cookie: sessionid=foo;another-cookie=bar
Host: pie.dev
User-Agent: HTTPie/0.9.9`

If you often deal with cookies in your requests, then you’d appreciate the [sessions][100] feature.

## [Authentication][101]

The currently supported authentication schemes are Basic and Digest (see [auth plugins][102] for
more). There are two flags that control authentication:

───┬────────────────────────────────────────────────────────────────────────────────────────────────
Fla│Arguments                                                                                       
g  │                                                                                                
───┼────────────────────────────────────────────────────────────────────────────────────────────────
`--│Pass either a `username:password` pair or a `token` as the argument. If the selected            
aut│authenticated method requires username/password combination and if you only specify a username  
h, │(`-a username`), you’ll be prompted for the password before the request is sent. To send an     
-a`│empty password, pass `username:`. The `username:password@hostname` URL syntax is supported as   
   │well (but credentials passed via `-a` have higher priority)                                     
───┼────────────────────────────────────────────────────────────────────────────────────────────────
`--│Specify the auth mechanism. Possible values are `basic`, `digest`, `bearer` or the name of any  
aut│[auth plugins][103] you have installed. The default value is `basic` so it can often be omitted 
h-t│                                                                                                
ype│                                                                                                
,  │                                                                                                
-A`│                                                                                                
───┴────────────────────────────────────────────────────────────────────────────────────────────────

### [Basic auth][104]

`$ http -a username:password pie.dev/basic-auth/username/password`
Run

### [Digest auth][105]

`$ http -A digest -a username:password pie.dev/digest-auth/httpie/username/password`
Run

### [Bearer auth][106]

`https -A bearer -a token pie.dev/bearer`
Run

### [Password prompt][107]

If you omit the password part of `--auth, -a`, HTTPie securely prompts you for it:

`$ http -a username pie.dev/basic-auth/username/password`
Run

Please note that when you use [`--session`][108], prompted passwords are persisted in session files.

### [Empty password][109]

To send an empty password without being prompted for it, include a trailing colon in the
credentials:

`$ http -a username: pie.dev/headers`
Run

### [`.netrc`][110]

Authentication information from your `~/.netrc` file is by default honored as well.

For example:

`$ cat ~/.netrc
machine pie.dev
login httpie
password test`
`$ http pie.dev/basic-auth/httpie/test
HTTP/1.1 200 OK
[...]`

This can be disabled with the `--ignore-netrc` option:

`$ http --ignore-netrc pie.dev/basic-auth/httpie/test
HTTP/1.1 401 UNAUTHORIZED
[...]`

### [Auth plugins][111]

Additional authentication mechanism can be installed as plugins. They can be found on the [Python
Package Index][112]. Here are a few picks:

* [httpie-api-auth][113]: ApiAuth
* [httpie-aws-auth][114]: AWS / Amazon S3
* [httpie-edgegrid][115]: EdgeGrid
* [httpie-hmac-auth][116]: HMAC
* [httpie-jwt-auth][117]: JWTAuth (JSON Web Tokens)
* [httpie-negotiate][118]: SPNEGO (GSS Negotiate)
* [httpie-ntlm][119]: NTLM (NT LAN Manager)
* [httpie-oauth1][120]: OAuth 1.0a
* [requests-hawk][121]: Hawk

See [plugin manager][122] for more details.

## [HTTP redirects][123]

By default, HTTP redirects are not followed and only the first response is shown:

`$ http pie.dev/redirect/3`
Run

### [Follow `Location`][124]

To instruct HTTPie to follow the `Location` header of `30x` responses and show the final response
instead, use the `--follow, -F` option:

`$ http --follow pie.dev/redirect/3`
Run

With `307 Temporary Redirect` and `308 Permanent Redirect`, the method and the body of the original
request are reused to perform the redirected request. Otherwise, a body-less `GET` request is
performed.

### [Showing intermediary redirect responses][125]

If you wish to see the intermediary requests/responses, then use the `--all` option:

`$ http --follow --all pie.dev/redirect/3`
Run

### [Limiting maximum redirects followed][126]

To change the default limit of maximum `30` redirects, use the `--max-redirects=<limit>` option:

`$ http --follow --all --max-redirects=2 pie.dev/redirect/3`
Run

## [Proxies][127]

You can specify proxies to be used through the `--proxy` argument for each protocol (which is
included in the value in case of redirects across protocols):

`$ http --proxy=http:http://10.10.1.10:3128 --proxy=https:https://10.10.1.10:1080 example.org`
Run

With Basic authentication:

`$ http --proxy=http:http://user:[[email protected]][128]:3128 example.org`
Run

### [Environment variables][129]

You can also configure proxies by environment variables `ALL_PROXY`, `HTTP_PROXY` and `HTTPS_PROXY`,
and the underlying [Requests library][130] will pick them up. If you want to disable proxies
configured through the environment variables for certain hosts, you can specify them in `NO_PROXY`.

In your `~/.bash_profile`:

`export HTTP_PROXY=http://10.10.1.10:3128
export HTTPS_PROXY=https://10.10.1.10:1080
export NO_PROXY=localhost,example.com`

### [SOCKS][131]

Usage for SOCKS is the same as for other types of [proxies][132]:

`$ http --proxy=http:socks5://user:pass@host:port --proxy=https:socks5://user:pass@host:port example
.org`
Run

## [HTTPS][133]

### [Server SSL certificate verification][134]

To skip the host’s SSL certificate verification, you can pass `--verify=no` (default is `yes`):

`$ http --verify=no https://pie.dev/get`
Run

### [Custom CA bundle][135]

You can also use `--verify=<CA_BUNDLE_PATH>` to set a custom CA bundle path:

`$ http --verify=/ssl/custom_ca_bundle https://example.org`
Run

### [Client side SSL certificate][136]

To use a client side certificate for the SSL communication, you can pass the path of the cert file
with `--cert`:

`$ http --cert=client.pem https://example.org`
Run

If the private key is not contained in the cert file, you may pass the path of the key file with
`--cert-key`:

`$ http --cert=client.crt --cert-key=client.key https://example.org`
Run

If the given private key requires a passphrase, HTTPie will automatically detect it and ask it
through a prompt:

`$ http --cert=client.pem --cert-key=client.key https://example.org
http: passphrase for client.key: ****`

If you don't want to see a prompt, you can supply the passphrase with the `--cert-key-pass`
argument:

`$ http --cert=client.pem --cert-key=client.key --cert-key-pass=my_password https://example.org`
Run

### [SSL version][137]

Use the `--ssl=<PROTOCOL>` option to specify the desired protocol version to use. This will default
to SSL v2.3 which will negotiate the highest protocol that both the server and your installation of
OpenSSL support. The available protocols are `ssl2.3`, `ssl3`, `tls1`, `tls1.1`, `tls1.2`, `tls1.3`.
(The actually available set of protocols may vary depending on your OpenSSL installation.)

`# Specify the vulnerable SSL v3 protocol to talk to an outdated server:
$ http --ssl=ssl3 https://vulnerable.example.org`
Run

### [SSL ciphers][138]

You can specify the available ciphers with `--ciphers`. It should be a string in the [OpenSSL cipher
list format][139].

`$ http --ciphers=ECDHE-RSA-AES128-GCM-SHA256 https://pie.dev/get`
Run

Note: these cipher strings do not change the negotiated version of SSL or TLS, they only affect the
list of available cipher suites.

To see the default cipher string, run `http --help` and see the `--ciphers` section under SSL.

## [Output options][140]

By default, HTTPie only outputs the final response and the whole response message is printed
(headers as well as the body). You can control what should be printed via several options:

────────────────────┬───────────────────────────────────────────────────────────────────────────────
Option              │What is printed                                                                
────────────────────┼───────────────────────────────────────────────────────────────────────────────
`--headers, -h`     │Only the response headers are printed                                          
────────────────────┼───────────────────────────────────────────────────────────────────────────────
`--body, -b`        │Only the response body is printed                                              
────────────────────┼───────────────────────────────────────────────────────────────────────────────
`--meta, -m`        │Only the [response metadata][141] is printed                                   
────────────────────┼───────────────────────────────────────────────────────────────────────────────
`--verbose, -v`     │Print the whole HTTP exchange (request and response). This option also enables 
                    │`--all` (see below)                                                            
────────────────────┼───────────────────────────────────────────────────────────────────────────────
`--verbose          │Just like `-v`, but also include the response metadata.                        
--verbose, -vv`     │                                                                               
────────────────────┼───────────────────────────────────────────────────────────────────────────────
`--print, -p`       │Selects parts of the HTTP exchange                                             
────────────────────┼───────────────────────────────────────────────────────────────────────────────
`--quiet, -q`       │Don’t print anything to `stdout` and `stderr`                                  
────────────────────┴───────────────────────────────────────────────────────────────────────────────

### [What parts of the HTTP exchange should be printed][142]

All the other [output options][143] are under the hood just shortcuts for the more powerful
`--print, -p`. It accepts a string of characters each of which represents a specific part of the
HTTP exchange:

─────────┬──────────────────
Character│Stands for        
─────────┼──────────────────
`H`      │request headers   
─────────┼──────────────────
`B`      │request body      
─────────┼──────────────────
`h`      │response headers  
─────────┼──────────────────
`b`      │response body     
─────────┼──────────────────
`m`      │[response         
         │meta][144]        
─────────┴──────────────────

Print request and response headers:

`$ http --print=Hh PUT pie.dev/put hello=world`
Run

#### [Response meta][145]

The response metadata section currently includes the total time elapsed. It’s the number of seconds
between opening the network connection and downloading the last byte of response the body.

To *only* show the response metadata, use `--meta, -m` (analogically to `--headers, -h` and `--body,
-b`):

`$ http --meta pie.dev/delay/1`
Run
`Elapsed time: 1.099171542s`

The [extra verbose `-vv` output][146] includes the meta section by default. You can also show it in
combination with other parts of the exchange via [`--print=m`][147]. For example, here we print it
together with the response headers:

`$ http --print=hm pie.dev/get`
Run
`HTTP/1.1 200 OK
Content-Type: application/json

Elapsed time: 0.077538375s`

Please note that it also includes time spent on formatting the output, which adds a small penalty.
Also, if the body is not part of the output, [we don’t spend time downloading it][148].

If you [use `--style` with one of the Pie themes][149], you’ll see the time information color-coded
(green/yellow/orange/red) based on how long the exchange took.

### [Verbose output][150]

`--verbose` can often be useful for debugging the request and generating documentation examples:

`$ http --verbose PUT pie.dev/put hello=world
PUT /put HTTP/1.1
Accept: application/json, */*;q=0.5
Accept-Encoding: gzip, deflate
Content-Type: application/json
Host: pie.dev
User-Agent: HTTPie/0.2.7dev

{
    "hello": "world"
}

HTTP/1.1 200 OK
Connection: keep-alive
Content-Length: 477
Content-Type: application/json
Date: Sun, 05 Aug 2012 00:25:23 GMT
Server: gunicorn/0.13.4

{
    […]
}`

#### [Extra verbose output][151]

If you run HTTPie with `-vv` or `--verbose --verbose`, then it would also display the [response
metadata][152].

`# Just like the above, but with additional columns like the total elapsed time
$ http -vv pie.dev/get`
Run

### [Quiet output][153]

`--quiet` redirects all output that would otherwise go to `stdout` and `stderr` to `/dev/null`
(except for errors and warnings). This doesn’t affect output to a file via `--output` or
`--download`.

`# There will be no output:
$ http --quiet pie.dev/post enjoy='the silence'`
Run

If you’d like to silence warnings as well, use `-q` or `--quiet` twice:

`# There will be no output, even in case of an unexpected response status code:
$ http -qq --check-status pie.dev/post enjoy='the silence without warnings'`
Run

### [Update warnings][154]

When there is a new release available for your platform (for example; if you installed HTTPie
through `pip`, it will check the latest version on `PyPI`), HTTPie will regularly warn you about the
new update (once a week). If you want to disable this behavior, you can set
`disable_update_warnings` to `true` in your [config][155] file.

### [Viewing intermediary requests/responses][156]

To see all the HTTP communication, i.e. the final request/response as well as any possible
intermediary requests/responses, use the `--all` option. The intermediary HTTP communication include
followed redirects (with `--follow`), the first unauthorized request when HTTP digest authentication
is used (`--auth=digest`), etc.

`# Include all responses that lead to the final one:
$ http --all --follow pie.dev/redirect/3`
Run

The intermediary requests/responses are by default formatted according to `--print, -p` (and its
shortcuts described above).

### [Conditional body download][157]

As an optimization, the response body is downloaded from the server only if it’s part of the output.
This is similar to performing a `HEAD` request, except that it applies to any HTTP method you use.

Let’s say that there is an API that returns the whole resource when it is updated, but you are only
interested in the response headers to see the status code after an update:

`$ http --headers PATCH pie.dev/patch name='New Name'`
Run

Since you are only printing the HTTP headers here, the connection to the server is closed as soon as
all the response headers have been received. Therefore, bandwidth and time isn’t wasted downloading
the body which you don’t care about. The response headers are downloaded always, even if they are
not part of the output

## [Raw request body][158]

In addition to crafting structured [JSON][159] and [forms][160] requests with the [request
items][161] syntax, you can provide a raw request body that will be sent without further processing.
These two approaches for specifying request data (i.e., structured and raw) cannot be combined.

There are three methods for passing raw request data: piping via `stdin`, `--raw='data'`, and
`@/file/path`.

### [Redirected Input][162]

The universal method for passing request data is through redirected `stdin` (standard input)—piping.

By default, `stdin` data is buffered and then with no further processing used as the request body.
If you provide `Content-Length`, then the request body is streamed without buffering. You may also
use `--chunked` to enable streaming via [chunked transfer encoding][163] or `--compress, -x` to
[compress the request body][164].

There are multiple useful ways to use piping:

Redirect from a file:

`$ http PUT pie.dev/put X-API-Token:123 < files/data.json`
Run

Or the output of another program:

`$ grep '401 Unauthorized' /var/log/httpd/error_log | http POST pie.dev/post`
Run

You can use `echo` for simple data:

`$ echo -n '{"name": "John"}' | http PATCH pie.dev/patch X-API-Token:123`
Run

You can also use a Bash *here string*:

`$ http pie.dev/post <<<'{"name": "John"}'`
Run

You can even pipe web services together using HTTPie:

`$ http GET https://api.github.com/repos/httpie/cli | http POST pie.dev/post`
Run

You can use `cat` to enter multiline data on the terminal:

`$ cat | http POST pie.dev/post
<paste>
^D`
`$ cat | http POST pie.dev/post Content-Type:text/plain
- buy milk
- call parents
^D`

On macOS, you can send the contents of the clipboard with `pbpaste`:

`$ pbpaste | http PUT pie.dev/put`
Run

Passing data through `stdin` **can’t** be combined with data fields specified on the command line:

`$ echo -n 'data' | http POST example.org more=data  # This is invalid`
Run

To prevent HTTPie from reading `stdin` data you can use the `--ignore-stdin` option.

### [Request data via `--raw`][165]

In a situation when piping data via `stdin` is not convenient (for example, when generating API docs
examples), you can specify the raw request body via the `--raw` option.

`$ http --raw 'Hello, world!' pie.dev/post`
Run
`$ http --raw '{"name": "John"}' pie.dev/post`
Run

### [Request data from a filename][166]

An alternative to redirected `stdin` is specifying a filename (as `@/path/to/file`) whose content is
used as if it came from `stdin`.

It has the advantage that the `Content-Type` header is automatically set to the appropriate value
based on the filename extension. For example, the following request sends the verbatim contents of
that XML file with `Content-Type: application/xml`:

`$ http PUT pie.dev/put @files/data.xml`
Run

File uploads are always streamed to avoid memory issues with large files.

## [Chunked transfer encoding][167]

You can use the `--chunked` flag to instruct HTTPie to use `Transfer-Encoding: chunked`:

`$ http --chunked PUT pie.dev/put hello=world`
Run
`$ http --chunked --multipart PUT pie.dev/put hello=world foo@files/data.xml`
Run
`$ http --chunked pie.dev/post @files/data.xml`
Run
`$ cat files/data.xml | http --chunked pie.dev/post`
Run

## [Compressed request body][168]

You can use the `--compress, -x` flag to instruct HTTPie to use `Content-Encoding: deflate` and
compress the request data:

`$ http --compress pie.dev/post @files/data.xml`
Run
`$ cat files/data.xml | http --compress pie.dev/post`
Run

If compressing the data does not save size, HTTPie sends it untouched. To always compress the data,
specify `--compress, -x` twice:

`$ http -xx PUT pie.dev/put hello=world`
Run

## [Terminal output][169]

HTTPie does several things by default in order to make its terminal output easy to read.

### [Colors and formatting][170]

Syntax highlighting is applied to HTTP headers and bodies (where it makes sense). You can choose
your preferred color scheme via the `--style` option if you don’t like the default one. There are
dozens of styles available, here are just a few notable ones:

───┬────────────────────────────────────────────────────────────────────────────────────────────────
Sty│Description                                                                                     
le │                                                                                                
───┼────────────────────────────────────────────────────────────────────────────────────────────────
`au│Follows your terminal ANSI color styles. This is the default style used by HTTPie               
to`│                                                                                                
───┼────────────────────────────────────────────────────────────────────────────────────────────────
`de│Default styles of the underlying Pygments library. Not actually used by default by HTTPie. You  
fau│can enable it with `--style=default`                                                            
lt`│                                                                                                
───┼────────────────────────────────────────────────────────────────────────────────────────────────
`pi│HTTPie’s original brand style. Also used in [HTTPie for Web and Desktop][171].                  
e-d│                                                                                                
ark│                                                                                                
`  │                                                                                                
───┼────────────────────────────────────────────────────────────────────────────────────────────────
`pi│Like `pie-dark`, but for terminals with light background colors.                                
e-l│                                                                                                
igh│                                                                                                
t` │                                                                                                
───┼────────────────────────────────────────────────────────────────────────────────────────────────
`pi│A generic version of `pie-dark` and `pie-light` themes that can work with any terminal          
e` │background. Its universality requires compromises in terms of legibility, but it’s useful if you
   │frequently switch your terminal between dark and light backgrounds.                             
───┼────────────────────────────────────────────────────────────────────────────────────────────────
`mo│A popular color scheme. Enable with `--style=monokai`                                           
nok│                                                                                                
ai`│                                                                                                
───┼────────────────────────────────────────────────────────────────────────────────────────────────
`fr│A bold, colorful scheme. Enable with `--style=fruity`                                           
uit│                                                                                                
y` │                                                                                                
───┼────────────────────────────────────────────────────────────────────────────────────────────────
…  │See `$ http --help` for all the possible `--style` values                                       
───┴────────────────────────────────────────────────────────────────────────────────────────────────

Use one of these options to control output processing:

───────────────┬─────────────────────────────────────────────────────────────
Option         │Description                                                  
───────────────┼─────────────────────────────────────────────────────────────
`--pretty=all` │Apply both colors and formatting. Default for terminal output
───────────────┼─────────────────────────────────────────────────────────────
`--pretty=color│Apply colors                                                 
s`             │                                                             
───────────────┼─────────────────────────────────────────────────────────────
`--pretty=forma│Apply formatting                                             
t`             │                                                             
───────────────┼─────────────────────────────────────────────────────────────
`--pretty=none`│Disables output processing. Default for redirected output    
───────────────┴─────────────────────────────────────────────────────────────

HTTPie looks at `Content-Type` to select the right syntax highlighter and formatter for each message
body. If that fails (e.g., the server provides the wrong type), or you prefer a different treatment,
you can manually overwrite the mime type for a response with `--response-mime`:

`$ http --response-mime=text/yaml pie.dev/get`
Run

Formatting has the following effects:

* HTTP headers are sorted by name.
* JSON data is indented, sorted by keys, and unicode escapes are converted to the characters they
  represent.
* XML and XHTML data is indented.

Please note that sometimes there might be changes made by formatters on the actual response body
(e.g., collapsing empty tags on XML) but the end result will always be semantically
indistinguishable. Some of these formatting changes can be configured more granularly through
[format options][172].

### [Format options][173]

The `--format-options=opt1:value,opt2:value` option allows you to control how the output should be
formatted when formatting is applied. The following options are available:

──────────────┬─────────────┬───────────────────
Option        │Default value│Shortcuts          
──────────────┼─────────────┼───────────────────
`headers.sort`│`true`       │`--sorted`,        
              │             │`--unsorted`       
──────────────┼─────────────┼───────────────────
`json.format` │`true`       │N/A                
──────────────┼─────────────┼───────────────────
`json.indent` │`4`          │N/A                
──────────────┼─────────────┼───────────────────
`json.sort_key│`true`       │`--sorted`,        
s`            │             │`--unsorted`       
──────────────┼─────────────┼───────────────────
`xml.format`  │`true`       │N/A                
──────────────┼─────────────┼───────────────────
`xml.indent`  │`2`          │N/A                
──────────────┴─────────────┴───────────────────

For example, this is how you would disable the default header and JSON key sorting, and specify a
custom JSON indent size:

`$ http --format-options headers.sort:false,json.sort_keys:false,json.indent:2 pie.dev/get`
Run

There are also two shortcuts that allow you to quickly disable and re-enable sorting-related format
options (currently it means JSON keys and headers): `--unsorted` and `--sorted`.

This is something you will typically store as one of the default options in your [config][174] file.

### [Redirected output][175]

HTTPie uses a different set of defaults for redirected output than for [terminal output][176]. The
differences being:

* Formatting and colors aren’t applied (unless `--pretty` is specified).
* Only the response body is printed (unless one of the [output options][177] is set).
* Also, binary data isn’t suppressed.

The reason is to make piping HTTPie’s output to another programs and downloading files work with no
extra flags. Most of the time, only the raw response body is of an interest when the output is
redirected.

Download a file:

`$ http pie.dev/image/png > image.png`
Run

Download an image of an [Octocat][178], resize it using [ImageMagick][179], and upload it elsewhere:

`$ http octodex.github.com/images/original.jpg | convert - -resize 25% - | http example.org/Octocats
`
Run

Force colorizing and formatting, and show both the request and the response in `less` pager:

`$ http --pretty=all --verbose pie.dev/get | less -R`
Run

The `-R` flag tells `less` to interpret color escape sequences included HTTPie’s output.

You can create a shortcut for invoking HTTPie with colorized and paged output by adding the
following to your `~/.bash_profile`:

`function httpless {
    # `httpless example.org'
    http --pretty=all --print=hb "$@" | less -R;
}`

### [Binary data][180]

Binary data is suppressed for terminal output, which makes it safe to perform requests to URLs that
send back binary data. Binary data is also suppressed in redirected but prettified output. The
connection is closed as soon as we know that the response body is binary,

`$ http pie.dev/bytes/2000`
Run

You will nearly instantly see something like this:

`HTTP/1.1 200 OK
Content-Type: application/octet-stream

+-----------------------------------------+
| NOTE: binary data not shown in terminal |
+-----------------------------------------+`

### [Display encoding][181]

HTTPie tries to do its best to decode message bodies when printing them to the terminal correctly.
It uses the encoding specified in the `Content-Type` `charset` attribute. If a message doesn’t
define its charset, we auto-detect it. For very short messages (1–32B), where auto-detection would
be unreliable, we default to UTF-8. For cases when the response encoding is still incorrect, you can
manually overwrite the response charset with `--response-charset`:

`$ http --response-charset=big5 pie.dev/get`
Run

## [Download mode][182]

HTTPie features a download mode in which it acts similarly to `wget`.

When enabled using the `--download, -d` flag, response headers are printed to the terminal
(`stderr`), and a progress bar is shown while the response body is being saved to a file.

`$ http --download https://github.com/httpie/cli/archive/master.tar.gz`
Run
`HTTP/1.1 200 OK
Content-Disposition: attachment; filename=httpie-master.tar.gz
Content-Length: 257336
Content-Type: application/x-gzip

Downloading 251.30 kB to "httpie-master.tar.gz"
Done. 251.30 kB in 2.73862s (91.76 kB/s)`

### [Downloaded filename][183]

There are three mutually exclusive ways through which HTTPie determines the output filename (with
decreasing priority):

1. You can explicitly provide it via `--output, -o`. The file gets overwritten if it already exists
   (or appended to with `--continue, -c`).
2. The server may specify the filename in the optional `Content-Disposition` response header. Any
   leading dots are stripped from a server-provided filename.
3. The last resort HTTPie uses is to generate the filename from a combination of the request URL and
   the response `Content-Type`. The initial URL is always used as the basis for the generated
   filename — even if there has been one or more redirects.

To prevent data loss by overwriting, HTTPie adds a unique numerical suffix to the filename when
necessary (unless specified with `--output, -o`).

### [Piping while downloading][184]

You can also redirect the response body to another program while the response headers and progress
are still shown in the terminal:

`$ http -d https://github.com/httpie/cli/archive/master.tar.gz | tar zxf -`
Run

### [Resuming downloads][185]

If `--output, -o` is specified, you can resume a partial download using the `--continue, -c` option.
This only works with servers that support `Range` requests and `206 Partial Content` responses. If
the server doesn’t support that, the whole file will simply be downloaded:

`$ http -dco file.zip example.org/file`
Run

`-dco` is shorthand for `--download` `--continue` `--output`.

### [Other notes][186]

* The `--download` option only changes how the response body is treated.
* You can still set custom headers, use sessions, `--verbose, -v`, etc.
* `--download` always implies `--follow` (redirects are followed).
* `--download` also implies `--check-status` (error HTTP status will result in a non-zero exist
  static code).
* HTTPie exits with status code `1` (error) if the body hasn’t been fully downloaded.
* `Accept-Encoding` can’t be set with `--download`.

## [Streamed responses][187]

Responses are downloaded and printed in chunks. This allows for streaming and large file downloads
without using too much memory. However, when [colors and formatting][188] are applied, the whole
response is buffered and only then processed at once.

### [Disabling buffering][189]

You can use the `--stream, -S` flag to make two things happen:

1. The output is flushed in much smaller chunks without any buffering, which makes HTTPie behave
   kind of like `tail -f` for URLs.
2. Streaming becomes enabled even when the output is prettified: It will be applied to each line of
   the response and flushed immediately. This makes it possible to have a nice output for long-lived
   requests, such as one to the [Twitter streaming API][190].

The `--stream` option is automatically enabled when the response headers include `Content-Type:
text/event-stream`.

### [Example use cases][191]

Prettified streamed response:

`$ http --stream pie.dev/stream/3`
Run

Streamed output by small chunks à la `tail -f`:

`# Send each new line (JSON object) to another URL as soon as it arrives from a streaming API:
$ http --stream pie.dev/stream/3 | while read line; do echo "$line" | http pie.dev/post ; done`
Run

## [Sessions][192]

By default, every request HTTPie makes is completely independent of any previous ones to the same
host.

However, HTTPie also supports persistent sessions via the `--session=SESSION_NAME_OR_PATH` option.
In a session, custom [HTTP headers][193] (except for the ones starting with `Content-` or `If-`),
[authentication][194], and [cookies][195] (manually specified or sent by the server) persist between
requests to the same host.

`# Create a new session:
$ http --session=./session.json pie.dev/headers API-Token:123`
Run
`# Inspect / edit the generated session file:
$ cat session.json`
Run
`# Re-use the existing session — the API-Token header will be set:
$ http --session=./session.json pie.dev/headers`
Run

All session data, including credentials, prompted passwords, cookie data, and custom headers are
stored in plain text. That means session files can also be created and edited manually in a text
editor—they are regular JSON. It also means that they can be read by anyone who has access to the
session file.

### [Named sessions][196]

You can create one or more named session per host. For example, this is how you can create a new
session named `user1` for `pie.dev`:

`$ http --session=user1 -a user1:password pie.dev/get X-Foo:Bar`
Run

From now on, you can refer to the session by its name (`user1`). When you choose to use the session
again, all previously specified authentication or HTTP headers will automatically be set:

`$ http --session=user1 pie.dev/get`
Run

To create or reuse a different session, simply specify a different name:

`$ http --session=user2 -a user2:password pie.dev/get X-Bar:Foo`
Run

Named sessions’ data is stored in JSON files inside the `sessions` subdirectory of the [config][197]
directory, typically `~/.config/httpie/sessions/<host>/<name>.json`
(`%APPDATA%\httpie\sessions\<host>\<name>.json` on Windows).

If you have executed the above commands on a Unix machine, you should be able to list the generated
sessions files using:

`$ ls -l ~/.config/httpie/sessions/pie.dev`
Run

### [Anonymous sessions][198]

Instead of giving it a name, you can also directly specify a path to a session file. This allows for
sessions to be re-used across multiple hosts:

`# Create a session:
$ http --session=/tmp/session.json example.org`
Run
`# Use the session to make a request to another host:
$ http --session=/tmp/session.json admin.example.org`
Run
`# You can also refer to a previously created named session:
$ http --session=~/.config/httpie/sessions/another.example.org/test.json example.org`
Run

When creating anonymous sessions, please remember to always include at least one `/`, even if the
session files is located in the current directory (i.e. `--session=./session.json` instead of just
`--session=session.json`), otherwise HTTPie assumes a named session instead.

### [Readonly session][199]

To use the original session file without updating it from the request/response exchange after it has
been created, specify the session name via `--session-read-only=SESSION_NAME_OR_PATH` instead.

`# If the session file doesn’t exist, then it is created:
$ http --session-read-only=./ro-session.json pie.dev/headers Custom-Header:orig-value`
Run
`# But it is not updated:
$ http --session-read-only=./ro-session.json pie.dev/headers Custom-Header:new-value`
Run

### [Host-based cookie policy][200]

Cookies persisted in sessions files have a `domain` field. This *binds* them to a specified
hostname. For example:

`{
    "cookies": [
        {
            "domain": "pie.dev",
            "name": "pie",
            "value": "apple"
        },
        {
            "domain": "httpbin.org",
            "name": "bin",
            "value": "http"
        }
    ]
}`

Using this session file, we include `Cookie: pie=apple` only in requests against `pie.dev` and
subdomains (e.g., `foo.pie.dev` or `foo.bar.pie.dev`):

`$ http --session=./session.json pie.dev/cookies`
Run
`{
    "cookies": {
        "pie": "apple"
    }
}`

To make a cookie domain *unbound* (i.e., to make it available to all hosts, including throughout a
cross-domain redirect chain), you can set the `domain` field to `null` in the session file:

`{
    "cookies": [
        {
            "domain": null,
            "name": "unbound-cookie",
            "value": "send-me-to-any-host"
        }
    ]
}`
`$ http --session=./session.json pie.dev/cookies`
Run
`{
    "cookies": {
        "unbound-cookie": "send-me-to-any-host"
    }
}`

### [Cookie storage behavior][201]

There are three possible sources of persisted cookies within a session. They have the following
storage priority: 1—response; 2—command line; 3—session file.

1. Receive a response with a `Set-Cookie` header:
   
   `$ http --session=./session.json pie.dev/cookie/set?foo=bar`
   Run
2. Send a cookie specified on the command line as seen in [cookies][202]:
   
   `$ http --session=./session.json pie.dev/headers Cookie:foo=bar`
   Run
3. Manually set cookie parameters in the session file:
   
   `{
      "cookies": {
          "foo": {
              "expires": null,
              "path": "/",
              "secure": false,
              "value": "bar"
              }
      }
   }`

In summary:

* Cookies set via the CLI overwrite cookies of the same name inside session files.
* Server-sent `Set-Cookie` header cookies overwrite any pre-existing ones with the same name.

Cookie expiration handling:

* When the server expires an existing cookie, HTTPie removes it from the session file.
* When a cookie in a session file expires, HTTPie removes it before sending a new request.

### [Upgrading sessions][203]

HTTPie may introduce changes in the session file format. When HTTPie detects an obsolete format, it
shows a warning. You can upgrade your session files using the following commands:

Upgrade all existing [named sessions][204] inside the `sessions` subfolder of your [config
directory][205]:

`$ httpie cli sessions upgrade-all
Upgraded 'api_auth' @ 'pie.dev' to v3.1.0
Upgraded 'login_cookies' @ 'httpie.io' to v3.1.0`

Upgrading individual sessions requires you to specify the session's hostname. That allows HTTPie to
find the correct file in the case of name sessions. Additionally, it allows it to correctly bind
cookies when upgrading with [`--bind-cookies`][206].

Upgrade a single [named session][207]:

`$ httpie cli sessions upgrade pie.dev api_auth
Upgraded 'api_auth' @ 'pie.dev' to v3.1.0`

Upgrade a single [anonymous session][208] using a file path:

`$ httpie cli sessions upgrade pie.dev ./session.json
Upgraded 'session.json' @ 'pie.dev' to v3.1.0`

#### [Session upgrade options][209]

These flags are available for both `sessions upgrade` and `sessions upgrade-all`:

──────────────┬─────────────────────────────────────────────────────────────────────────────
Option        │Description                                                                  
──────────────┼─────────────────────────────────────────────────────────────────────────────
`--bind-cookie│Bind all previously [unbound cookies][210] to the session’s host             
s`            │([context][211]).                                                            
──────────────┴─────────────────────────────────────────────────────────────────────────────

## [Config][212]

HTTPie uses a simple `config.json` file. The file doesn’t exist by default, but you can create it
manually.

### [Config file directory][213]

To see the exact location for your installation, run `http --debug` and look for `config_dir` in the
output.

The default location of the configuration file on most platforms is
`$XDG_CONFIG_HOME/httpie/config.json` (defaulting to `~/.config/httpie/config.json`).

For backward compatibility, if the directory `~/.httpie` exists, the configuration file there will
be used instead.

On Windows, the config file is located at `%APPDATA%\httpie\config.json`.

The config directory can be changed by setting the `$HTTPIE_CONFIG_DIR` environment variable:

`$ export HTTPIE_CONFIG_DIR=/tmp/httpie
$ http pie.dev/get`

### [Configurable options][214]

Currently, HTTPie offers a single configurable option:

#### [`default_options`][215]

An `Array` (by default empty) of default options that should be applied to every invocation of
HTTPie.

For instance, you can use this config option to change your default color theme:

`$ cat ~/.config/httpie/config.json`
Run
`{
    "default_options": [
        "--style=fruity"
    ]
}`

Technically, it is possible to include any HTTPie options in there. However, it is not recommended
modifying the default behavior in a way that would break your compatibility with the wider world as
that may become confusing.

#### [`plugins_dir`][216]

The directory where the plugins will be installed. HTTPie needs to have read/write access on that
directory, since `httpie cli plugins install` will download new plugins to there. See [plugin
manager][217] for more information.

### [Un-setting previously specified options][218]

Default options from the config file, or specified any other way, can be unset for a particular
invocation via `--no-OPTION` arguments passed via the command line (e.g., `--no-style` or
`--no-session`).

## [Scripting][219]

When using HTTPie from shell scripts, it can be handy to set the `--check-status` flag. It instructs
HTTPie to exit with an error if the HTTP status is one of `3xx`, `4xx`, or `5xx`. The exit status
will be `3` (unless `--follow` is set), `4`, or `5`, respectively.

`#!/bin/bash

if http --check-status --ignore-stdin --timeout=2.5 HEAD pie.dev/get &> /dev/null; then
    echo 'OK!'
else
    case $? in
        2) echo 'Request timed out!' ;;
        3) echo 'Unexpected HTTP 3xx Redirection!' ;;
        4) echo 'HTTP 4xx Client Error!' ;;
        5) echo 'HTTP 5xx Server Error!' ;;
        6) echo 'Exceeded --max-redirects=<n> redirects!' ;;
        *) echo 'Other Error!' ;;
    esac
fi`

### [Best practices][220]

The default behavior of automatically reading `stdin` is typically not desirable during
non-interactive invocations. You most likely want to use the `--ignore-stdin` option to disable it.

It's important to note that without the `--ignore-stdin` option, HTTPie may appear to have stopped
working (hang). This happens because, in situations where HTTPie is invoked outside of an
interactive session, such as from a cron job, `stdin` is not connected to a terminal. This means
that the rules for [redirected input][221] will be followed. When `stdin` is redirected, HTTPie
assumes that the input will contain the request body, and it waits for the input to be provided.
But, since there is neither any input data nor an end-of-file (`EOF`) signal, HTTPie gets stuck. To
avoid this problem, the `--ignore-stdin` flag should be used in scripts, unless data is being piped
to HTTPie.

To prevent your program from becoming unresponsive when the server fails to respond, it's a good
idea to use the `--timeout` option to set a connection timeout limit.

## [Plugin manager][222]

HTTPie offers extensibility through a [plugin API][223], and there are dozens of plugins available
to try! They add things like new authentication methods ([akamai/httpie-edgegrid][224]), transport
mechanisms ([httpie/httpie-unixsocket][225]), message convertors ([banteg/httpie-image][226]), or
simply change how a response is formatted.

> Note: Plugins are usually made by our community members, and thus have no direct relationship with
> the HTTPie project. We do not control / review them at the moment, so use them at your own
> discretion.

For managing these plugins; starting with 3.0, we are offering a new plugin manager.

This command is currently in beta.

### [`httpie cli`][227]

#### [`httpie cli check-updates`][228]

You can check whether a new update is available for your system by running `httpie cli
check-updates`:

`$ httpie cli check-updates`

#### [`httpie cli export-args`][229]

`httpie cli export-args` command can expose the parser specification of `http`/`https` commands
(like an API definition) to outside tools so that they can use this to build better interactions
over them (e.g., offer auto-complete).

Available formats to export in include:

────┬───────────────────────────────────────────────────────────────────────────────────────────────
Form│Description                                                                                    
at  │                                                                                               
────┼───────────────────────────────────────────────────────────────────────────────────────────────
`jso│Export the parser spec in JSON. The schema includes a top-level `version` parameter which      
n`  │should be interpreted in [semver][230].                                                        
────┴───────────────────────────────────────────────────────────────────────────────────────────────

You can use any of these formats with `--format` parameter, but the default one is `json`.

`$ httpie cli export-args | jq '"Program: " + .spec.name + ", Version: " +  .version'
"Program: http, Version: 0.0.1a0"`

#### [`httpie cli plugins`][231]

`plugins` interface is a very simple plugin manager for installing, listing and uninstalling HTTPie
plugins.

In the past `pip` was used to install/uninstall plugins, but on some environments (e.g., brew
installed packages) it wasn’t working properly. The new interface is a very simple overlay on top of
`pip` to allow plugin installations on every installation method.

By default, the plugins (and their missing dependencies) will be stored under the configuration
directory, but this can be modified through `plugins_dir` variable on the config.

[`httpie cli plugins install`][232]

For installing plugins from [PyPI][233] or from local paths, `httpie cli plugins install` can be
used.

`$ httpie cli plugins install httpie-plugin
Installing httpie-plugin...
Successfully installed httpie-plugin-1.0.2`

> Tip: Generally HTTPie plugins start with `httpie-` prefix. Try searching for it on [PyPI][234] to
> find out all plugins from the community.

[`httpie cli plugins list`][235]

List all installed plugins.

`$ httpie cli plugins list
httpie_plugin (1.0.2)
  httpie_plugin (httpie.plugins.auth.v1)
httpie_plugin_2 (1.0.6)
  httpie_plugin_2 (httpie.plugins.auth.v1)
httpie_converter (1.0.0)
  httpie_iterm_converter (httpie.plugins.converter.v1)
  httpie_konsole_konverter (httpie.plugins.converter.v1)`
[`httpie cli plugins upgrade`][236]

For upgrading already installed plugins, use `httpie plugins upgrade`.

`$ httpie cli plugins upgrade httpie-plugin`
Run
[`httpie cli plugins uninstall`][237]

Uninstall plugins from the isolated plugins directory. If the plugin is not installed through
`httpie cli plugins install`, it won’t uninstall it.

`$ httpie cli plugins uninstall httpie-plugin`
Run

## [Meta][238]

### [Interface design][239]

The syntax of the command arguments closely correspond to the actual HTTP requests sent over the
wire. It has the advantage that it’s easy to remember and read. You can often translate an HTTP
request to an HTTPie argument list just by inlining the request elements. For example, compare this
HTTP request:

`POST /post HTTP/1.1
Host: pie.dev
X-API-Key: 123
User-Agent: Bacon/1.0
Content-Type: application/x-www-form-urlencoded

name=value&name2=value2`

with the HTTPie command that sends it:

`$ http -f POST pie.dev/post \
    X-API-Key:123 \
    User-Agent:Bacon/1.0 \
    name=value \
    name2=value2`
Run

Notice that both the order of elements and the syntax are very similar, and that only a small
portion of the command is used to control HTTPie and doesn’t directly correspond to any part of the
request (here, it’s only `-f` asking HTTPie to send a form request).

The two modes, `--pretty=all` (default for terminal) and `--pretty=none` (default for [redirected
output][240]), allow for both user-friendly interactive use and usage from scripts, where HTTPie
serves as a generic HTTP client.

In the future, the command line syntax and some of the `--OPTIONS` may change slightly, as HTTPie
improves and new features are added. All changes are recorded in the [change log][241].

### [Community and Support][242]

HTTPie has the following community channels:

* [GitHub Issues][243] for bug reports and feature requests
* [Discord server][244] to ask questions, discuss features, and for general API development
  discussion
* [StackOverflow][245] to ask questions (make sure to use the [httpie][246] tag)

### [Related projects][247]

#### [Dependencies][248]

Under the hood, HTTPie uses these two amazing libraries:

* [Requests][249] — Python HTTP library for humans
* [Pygments][250] — Python syntax highlighter

#### [HTTPie friends][251]

HTTPie plays exceptionally well with the following tools:

* [http-prompt][252] — an interactive shell for HTTPie featuring autocomplete and command syntax
  highlighting
* [jq][253] — CLI JSON processor that works great in conjunction with HTTPie

Helpers to convert from other client tools:

* [CurliPie][254] — library to convert cURL commands to HTTPie

#### [Alternatives][255]

* [httpcat][256] — a lower-level sister utility of HTTPie for constructing raw HTTP requests on the
  command line
* [curl][257] — a "Swiss knife" command line tool and an exceptional library for transferring data
  with URLs.

### [Contributing][258]

See [CONTRIBUTING][259].

### [Security policy][260]

See [github.com/httpie/cli/security/policy][261].

### [Change log][262]

See [CHANGELOG][263].

### [Artwork][264]

* [README Animation][265] by [Allen Smith][266].

### [Licence][267]

BSD-3-Clause: [LICENSE][268].

### [Authors][269]

[Jakub Roztocil][270] ([@jakubroztocil][271]) created HTTPie and [these fine people][272] have
contributed.

[1]: /docs/cli/main-features
[2]: /docs/cli/installation
[3]: /docs/cli#universal
[4]: /docs/cli#macos
[5]: /docs/cli#windows
[6]: /docs/cli#linux
[7]: /docs/cli#freebsd
[8]: /docs/cli/universal
[9]: /docs/cli/pypi
[10]: /docs/cli/macos
[11]: /docs/cli/homebrew
[12]: https://brew.sh/
[13]: https://docs.brew.sh/Installation
[14]: /docs/cli/macports
[15]: https://www.macports.org/
[16]: https://www.macports.org/install.php
[17]: /docs/cli/windows
[18]: /docs/cli/chocolatey
[19]: https://chocolatey.org/
[20]: https://chocolatey.org/install
[21]: /docs/cli/linux
[22]: /docs/cli/debian-and-ubuntu
[23]: /docs/cli/fedora
[24]: /docs/cli/centos-and-rhel
[25]: /docs/cli/single-binary-executables
[26]: /docs/cli/snapcraft-linux
[27]: https://snapcraft.io/
[28]: https://snapcraft.io/docs/installing-snapd
[29]: /docs/cli/linuxbrew
[30]: https://docs.brew.sh/Homebrew-on-Linux
[31]: https://docs.brew.sh/Homebrew-on-Linux#install
[32]: /docs/cli/arch-linux
[33]: /docs/cli/freebsd
[34]: /docs/cli/freshports
[35]: /docs/cli/unstable-version
[36]: https://github.com/httpie/cli/blob/master/httpie/__init__.py#L6
[37]: /docs/cli/usage
[38]: /docs/cli/examples
[39]: /docs/cli#http-method
[40]: /docs/cli#http-headers
[41]: /docs/cli#json
[42]: /docs/cli#forms
[43]: /docs/cli#output-options
[44]: /docs/cli#offline-mode
[45]: https://developer.github.com/v3/issues/comments/#create-a-comment
[46]: https://github.com/httpie/cli/issues/83
[47]: /docs/cli#authentication
[48]: /docs/cli#redirected-input
[49]: /docs/cli#redirected-output
[50]: /docs/cli#sessions
[51]: /docs/cli/http-method
[52]: /docs/cli/optional-get-and-post
[53]: /docs/cli/request-url
[54]: /docs/cli/querystring-parameters
[55]: /docs/cli#file-based-separators
[56]: /docs/cli/url-shortcuts-for-localhost
[57]: /docs/cli/other-default-schemes
[58]: https://pypi.org/search/?q=httpie
[59]: https://github.com/httpie/httpie-unixsocket
[60]: /docs/cli/--path-as-is
[61]: /docs/cli/request-items
[62]: /cdn-cgi/l/email-protection
[63]: /docs/cli#raw-request-body
[64]: /docs/cli/file-based-separators
[65]: /docs/cli/escaping-rules
[66]: /docs/cli/json
[67]: /cdn-cgi/l/email-protection
[68]: /cdn-cgi/l/email-protection
[69]: /docs/cli/default-behavior
[70]: /docs/cli#request-items
[71]: /docs/cli/explicit-json
[72]: /docs/cli/non-string-json-fields
[73]: /docs/cli/nested-json
[74]: /docs/cli/introduction
[75]: /docs/cli/advanced-usage
[76]: /docs/cli/top-level-arrays
[77]: /docs/cli#raw-request-body
[78]: /docs/cli/escaping-behavior
[79]: /docs/cli#escaping-rules
[80]: /docs/cli/guiding-syntax-errors
[81]: /docs/cli/type-safety
[82]: /docs/cli/raw-json
[83]: /docs/cli#raw-request-body
[84]: /docs/cli/forms
[85]: /docs/cli#json
[86]: /docs/cli#config
[87]: /docs/cli/regular-forms
[88]: /docs/cli/file-upload-forms
[89]: /docs/cli/http-headers
[90]: /docs/cli/default-request-headers
[91]: /docs/cli/reading-headers-from-a-file
[92]: /docs/cli#file-based-separators
[93]: /docs/cli/empty-headers-and-header-un-setting
[94]: /docs/cli/multiple-header-values-with-the-same-name
[95]: /docs/cli/limiting-response-headers
[96]: /docs/cli/offline-mode
[97]: /docs/cli#output-options
[98]: /docs/cli/cookies
[99]: /docs/cli#http-headers
[100]: /docs/cli#sessions
[101]: /docs/cli/authentication
[102]: /docs/cli#auth-plugins
[103]: /docs/cli#auth-plugins
[104]: /docs/cli/basic-auth
[105]: /docs/cli/digest-auth
[106]: /docs/cli/bearer-auth
[107]: /docs/cli/password-prompt
[108]: /docs/cli#sessions
[109]: /docs/cli/empty-password
[110]: /docs/cli/netrc
[111]: /docs/cli/auth-plugins
[112]: https://pypi.python.org/pypi?%3Aaction=search&term=httpie&submit=search
[113]: https://github.com/pd/httpie-api-auth
[114]: https://github.com/httpie/httpie-aws-auth
[115]: https://github.com/akamai-open/httpie-edgegrid
[116]: https://github.com/guardian/httpie-hmac-auth
[117]: https://github.com/teracyhq/httpie-jwt-auth
[118]: https://github.com/ndzou/httpie-negotiate
[119]: https://github.com/httpie/httpie-ntlm
[120]: https://github.com/qcif/httpie-oauth1
[121]: https://github.com/mozilla-services/requests-hawk
[122]: /docs/cli#plugin-manager
[123]: /docs/cli/http-redirects
[124]: /docs/cli/follow-location
[125]: /docs/cli/showing-intermediary-redirect-responses
[126]: /docs/cli/limiting-maximum-redirects-followed
[127]: /docs/cli/proxies
[128]: /cdn-cgi/l/email-protection
[129]: /docs/cli/environment-variables
[130]: https://requests.readthedocs.io/en/latest/
[131]: /docs/cli/socks
[132]: /docs/cli#proxies
[133]: /docs/cli/https
[134]: /docs/cli/server-ssl-certificate-verification
[135]: /docs/cli/custom-ca-bundle
[136]: /docs/cli/client-side-ssl-certificate
[137]: /docs/cli/ssl-version
[138]: /docs/cli/ssl-ciphers
[139]: https://www.openssl.org/docs/man1.1.0/man1/ciphers.html
[140]: /docs/cli/output-options
[141]: /docs/cli#response-meta
[142]: /docs/cli/what-parts-of-the-http-exchange-should-be-printed
[143]: /docs/cli#output-options
[144]: /docs/cli#response-meta
[145]: /docs/cli/response-meta
[146]: /docs/cli#extra-verbose-output
[147]: /docs/cli#what-parts-of-the-http-exchange-should-be-printed
[148]: /docs/cli#conditional-body-download
[149]: /docs/cli#colors-and-formatting
[150]: /docs/cli/verbose-output
[151]: /docs/cli/extra-verbose-output
[152]: /docs/cli#response-meta
[153]: /docs/cli/quiet-output
[154]: /docs/cli/update-warnings
[155]: /docs/cli#config
[156]: /docs/cli/viewing-intermediary-requestsresponses
[157]: /docs/cli/conditional-body-download
[158]: /docs/cli/raw-request-body
[159]: /docs/cli#json
[160]: /docs/cli#forms
[161]: /docs/cli#request-items
[162]: /docs/cli/redirected-input
[163]: /docs/cli#chunked-transfer-encoding
[164]: /docs/cli#compressed-request-body
[165]: /docs/cli/request-data-via---raw
[166]: /docs/cli/request-data-from-a-filename
[167]: /docs/cli/chunked-transfer-encoding
[168]: /docs/cli/compressed-request-body
[169]: /docs/cli/terminal-output
[170]: /docs/cli/colors-and-formatting
[171]: https://httpie.io/product
[172]: /docs/cli#format-options
[173]: /docs/cli/format-options
[174]: /docs/cli#config
[175]: /docs/cli/redirected-output
[176]: /docs/cli#terminal-output
[177]: /docs/cli#output-options
[178]: https://octodex.github.com/images/original.jpg
[179]: https://imagemagick.org/
[180]: /docs/cli/binary-data
[181]: /docs/cli/display-encoding
[182]: /docs/cli/download-mode
[183]: /docs/cli/downloaded-filename
[184]: /docs/cli/piping-while-downloading
[185]: /docs/cli/resuming-downloads
[186]: /docs/cli/other-notes
[187]: /docs/cli/streamed-responses
[188]: /docs/cli#colors-and-formatting
[189]: /docs/cli/disabling-buffering
[190]: https://developer.twitter.com/en/docs/tutorials/consuming-streaming-data
[191]: /docs/cli/example-use-cases
[192]: /docs/cli/sessions
[193]: /docs/cli#http-headers
[194]: /docs/cli#authentication
[195]: /docs/cli#cookies
[196]: /docs/cli/named-sessions
[197]: /docs/cli#config
[198]: /docs/cli/anonymous-sessions
[199]: /docs/cli/readonly-session
[200]: /docs/cli/host-based-cookie-policy
[201]: /docs/cli/cookie-storage-behavior
[202]: /docs/cli#cookies
[203]: /docs/cli/upgrading-sessions
[204]: /docs/cli#named-sessions
[205]: https://httpie.io/docs/cli/config-file-directory
[206]: /docs/cli#session-upgrade-options
[207]: /docs/cli#named-sessions
[208]: /docs/cli#anonymous-sessions
[209]: /docs/cli/session-upgrade-options
[210]: /docs/cli#host-based-cookie-policy
[211]: https://github.com/httpie/cli/security/advisories/GHSA-9w4w-cpc8-h2fq
[212]: /docs/cli/config
[213]: /docs/cli/config-file-directory
[214]: /docs/cli/configurable-options
[215]: /docs/cli/default_options
[216]: /docs/cli/plugins_dir
[217]: /docs/cli#plugin-manager
[218]: /docs/cli/un-setting-previously-specified-options
[219]: /docs/cli/scripting
[220]: /docs/cli/best-practices
[221]: /docs/cli#redirected-input
[222]: /docs/cli/plugin-manager
[223]: https://github.com/httpie/cli/blob/master/httpie/plugins/base.py
[224]: https://github.com/akamai/httpie-edgegrid
[225]: https://github.com/httpie/httpie-unixsocket
[226]: https://github.com/banteg/httpie-image
[227]: /docs/cli/httpie-cli
[228]: /docs/cli/httpie-cli-check-updates
[229]: /docs/cli/httpie-cli-export-args
[230]: https://semver.org/
[231]: /docs/cli/httpie-cli-plugins
[232]: /docs/cli/httpie-cli-plugins-install
[233]: https://pypi.org/
[234]: https://pypi.org/search/?q=httpie-
[235]: /docs/cli/httpie-cli-plugins-list
[236]: /docs/cli/httpie-cli-plugins-upgrade
[237]: /docs/cli/httpie-cli-plugins-uninstall
[238]: /docs/cli/meta
[239]: /docs/cli/interface-design
[240]: /docs/cli#redirected-output
[241]: /docs/cli#change-log
[242]: /docs/cli/community-and-support
[243]: https://github.com/httpie/cli/issues
[244]: https://httpie.io/discord
[245]: https://stackoverflow.com
[246]: https://stackoverflow.com/questions/tagged/httpie
[247]: /docs/cli/related-projects
[248]: /docs/cli/dependencies
[249]: https://requests.readthedocs.io/en/latest/
[250]: https://pygments.org/
[251]: /docs/cli/httpie-friends
[252]: https://github.com/httpie/http-prompt
[253]: https://stedolan.github.io/jq/
[254]: https://curlipie.open-api.vn
[255]: /docs/cli/alternatives
[256]: https://github.com/httpie/httpcat
[257]: https://curl.haxx.se
[258]: /docs/cli/contributing
[259]: https://github.com/httpie/cli/blob/master/CONTRIBUTING.md
[260]: /docs/cli/security-policy
[261]: https://github.com/httpie/cli/security/policy
[262]: /docs/cli/change-log
[263]: https://github.com/httpie/cli/blob/master/CHANGELOG.md
[264]: /docs/cli/artwork
[265]: https://github.com/httpie/cli/blob/master/docs/httpie-animation.gif
[266]: https://github.com/loranallensmith
[267]: /docs/cli/licence
[268]: https://github.com/httpie/cli/blob/master/LICENSE
[269]: /docs/cli/authors
[270]: https://roztocil.co
[271]: https://twitter.com/jakubroztocil
[272]: https://github.com/httpie/cli/blob/master/AUTHORS.md
