[[Join the chat at https://kotlinlang.slack.com]][1] [[Build status]][2] [[Maven Central]][3]
[[JitPack]][4] [[HomeBrew]][5] [[License]][6] [[ktlint]][7]

[Kotlin][8] linter in spirit of [standard/standard][9] (JavaScript) and [gofmt][10] (Go).

## Key features

* No configuration required
* Built-in Rule sets
* Built-in formatter
* `.editorconfig` support
* Several built-in reporters: `plain`, `json`, `html` and `checkstyle`
* Executable jar
* Allows extension with custom rule sets and reporters

## Quick start

Follow steps below for a quick start with latest ktlint release.

* Step 1: Install with brew
  
  brew install ktlint
  
  See [download and verification from GitHub][11] or [other package managers][12] for alternative
  ways of installing ktlint. Or, use one of the [integrations like maven and gradle plugins][13].
* Step 2: Lint and format your code
  All files with extension `.kt` and `.kts` in the current directory and below will be scanned.
  Problems will be fixed automatically when possible.
  
  ktlint --format
  # or
  ktlint -F
  
  See [cli usage][14] for a more extensive description on using ktlint.

## Documentation

[User guide][15]

### Legal

This project is not affiliated with nor endorsed by JetBrains.
All code, unless specified otherwise, is licensed under the [MIT][16] license.
Copyright (c) 2019 Pinterest, Inc.
Copyright (c) 2016-2019 Stanley Shyiko.

[1]: https://kotlinlang.slack.com/messages/CKS3XG0LS
[2]: https://github.com/pinterest/ktlint/actions/workflows/publish-snapshot-build.yml
[3]: https://central.sonatype.com/artifact/com.pinterest.ktlint/ktlint-cli?smo=true
[4]: https://jitpack.io/#pinterest/ktlint
[5]: https://formulae.brew.sh/formula/ktlint
[6]: /pinterest/ktlint/blob/master/LICENSE
[7]: https://pinterest.github.io/ktlint/
[8]: https://kotlinlang.org/
[9]: https://github.com/standard/standard
[10]: https://golang.org/cmd/gofmt/
[11]: https://pinterest.github.io/ktlint/latest/install/cli/#download-and-verification
[12]: https://pinterest.github.io/ktlint/latest/install/cli/#package-managers
[13]: https://pinterest.github.io/ktlint/latest/install/integrations/
[14]: https://pinterest.github.io/ktlint/latest/install/cli/#command-line-usage
[15]: https://pinterest.github.io/ktlint/
[16]: https://opensource.org/licenses/MIT
