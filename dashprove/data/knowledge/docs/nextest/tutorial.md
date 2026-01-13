# Installing pre-built binaries[¶][1]

The quickest way to get going with nextest is to download a pre-built binary for your platform.

## Downloading and installing from your terminal[¶][2]

The instructions below are suitable for both end users and CI. These links will stay stable.

With cargo-binstall Linux x86_64 Linux aarch64 macOS universal Windows x86_64

If you have [cargo-binstall][3] available:

`cargo binstall cargo-nextest --secure
`

Info

The commands below assume that your Rust installation is managed via [rustup][4]. You can extract
the archive to a different directory in your PATH if required.

If you'd like to stay on the 0.9 series to avoid breaking changes (see the [stability policy][5] for
more), replace `latest` in the URL with `0.9`.

Run in a terminal:

`curl -LsSf https://get.nexte.st/latest/linux | tar zxf - -C ${CARGO_HOME:-~/.cargo}/bin
`

For a statically-linked binary with no runtime library dependencies, based on [musl][6]:

`curl -LsSf https://get.nexte.st/latest/linux-musl | tar zxf - -C ${CARGO_HOME:-~/.cargo}/bin
`

Info

The command below assumes that your Rust installation is managed via [rustup][7]. You can extract
the archive to a different directory in your PATH if required.

If you'd like to stay on the 0.9 series to avoid breaking changes (see the [stability policy][8] for
more), replace `latest` in the URL with `0.9`.

Run in a terminal:

`curl -LsSf https://get.nexte.st/latest/linux-arm | tar zxf - -C ${CARGO_HOME:-~/.cargo}/bin
`

Info

The command below assumes that your Rust installation is managed via [rustup][9]. You can extract
the archive to a different directory in your PATH if required.

If you'd like to stay on the 0.9 series to avoid breaking changes (see the [stability policy][10]
for more), replace `latest` in the URL with `0.9`.

Run in a terminal:

`curl -LsSf https://get.nexte.st/latest/mac | tar zxf - -C ${CARGO_HOME:-~/.cargo}/bin
`

This will download a universal binary that works on both Intel and Apple Silicon Macs.

Using WinGetPowerShellUnix shell

Run in a terminal:

`winget install nextest.cargo-nextest
`

Info

The commands below assume that your Rust installation is managed via [rustup][11]. You can extract
the archive to a different directory in your PATH if required.

If you'd like to stay on the 0.9 series to avoid breaking changes (see the [stability policy][12]
for more), replace `latest` in the URL with `0.9`.

Run in PowerShell:

`$tmp = New-TemporaryFile | Rename-Item -NewName { $_ -replace 'tmp$', 'zip' } -PassThru
Invoke-WebRequest -OutFile $tmp https://get.nexte.st/latest/windows
$outputDir = if ($Env:CARGO_HOME) { Join-Path $Env:CARGO_HOME "bin" } else { "~/.cargo/bin" }
$tmp | Expand-Archive -DestinationPath $outputDir -Force
$tmp | Remove-Item
`

Info

The commands below assume that your Rust installation is managed via [rustup][13]. You can extract
the archive to a different directory in your PATH if required.

If you'd like to stay on the 0.9 series to avoid breaking changes (see the [stability policy][14]
for more), replace `latest` in the URL with `0.9`.

Using a Unix shell, `curl`, and `tar` *natively* on Windows (e.g. `shell: bash` on GitHub Actions,
or Git Bash):

`curl -LsSf https://get.nexte.st/latest/windows-tar | tar zxf - -C ${CARGO_HOME:-~/.cargo}/bin
`

Windows Subsystem for Linux (WSL) users should follow the **Linux x86_64** instructions.

Other platforms
Windows aarch64 Windows i686 FreeBSD x86_64 illumos x86_64
Windows aarch64 using WinGetWindows aarch64 with PowerShellWindows aarch64 with a Unix shell

Run in a terminal:

`winget install nextest.cargo-nextest
`

Info

The commands below assume that your Rust installation is managed via [rustup][15]. You can extract
the archive to a different directory in your PATH if required.

If you'd like to stay on the 0.9 series to avoid breaking changes (see the [stability policy][16]
for more), replace `latest` in the URL with `0.9`.

Run in PowerShell:

`$tmp = New-TemporaryFile | Rename-Item -NewName { $_ -replace 'tmp$', 'zip' } -PassThru
Invoke-WebRequest -OutFile $tmp https://get.nexte.st/latest/windows-arm
$outputDir = if ($Env:CARGO_HOME) { Join-Path $Env:CARGO_HOME "bin" } else { "~/.cargo/bin" }
$tmp | Expand-Archive -DestinationPath $outputDir -Force
$tmp | Remove-Item
`

Info

The commands below assume that your Rust installation is managed via [rustup][17]. You can extract
the archive to a different directory in your PATH if required.

If you'd like to stay on the 0.9 series to avoid breaking changes (see the [stability policy][18]
for more), replace `latest` in the URL with `0.9`.

Using a Unix shell, `curl`, and `tar` *natively* on Windows (e.g. `shell: bash` on GitHub Actions,
or Git Bash):

`curl -LsSf https://get.nexte.st/latest/windows-arm-tar | tar zxf - -C ${CARGO_HOME:-~/.cargo}/bin
`
Windows i686 using WinGetWindows i686 with PowerShellWindows i686 with a Unix shell

Run in a terminal:

`winget install nextest.cargo-nextest
`

Info

The commands below assume that your Rust installation is managed via [rustup][19]. You can extract
the archive to a different directory in your PATH if required.

If you'd like to stay on the 0.9 series to avoid breaking changes (see the [stability policy][20]
for more), replace `latest` in the URL with `0.9`.

Run in PowerShell:

`$tmp = New-TemporaryFile | Rename-Item -NewName { $_ -replace 'tmp$', 'zip' } -PassThru
Invoke-WebRequest -OutFile $tmp https://get.nexte.st/latest/windows-x86
$outputDir = if ($Env:CARGO_HOME) { Join-Path $Env:CARGO_HOME "bin" } else { "~/.cargo/bin" }
$tmp | Expand-Archive -DestinationPath $outputDir -Force
$tmp | Remove-Item
`

Info

The commands below assume that your Rust installation is managed via [rustup][21]. You can extract
the archive to a different directory in your PATH if required.

If you'd like to stay on the 0.9 series to avoid breaking changes (see the [stability policy][22]
for more), replace `latest` in the URL with `0.9`.

Using a Unix shell, `curl`, and `tar` *natively* on Windows (e.g. `shell: bash` on GitHub Actions,
or Git Bash):

`curl -LsSf https://get.nexte.st/latest/windows-x86-tar | tar zxf - -C ${CARGO_HOME:-~/.cargo}/bin
`
`curl -LsSf https://get.nexte.st/latest/freebsd | tar zxf - -C ${CARGO_HOME:-~/.cargo}/bin
`
`curl -LsSf https://get.nexte.st/latest/illumos | gunzip | tar xf - -C ${CARGO_HOME:-~/.cargo}/bin
`

As of 2025-04, the current version of illumos tar has [a bug][23] where `tar zxf` doesn't work over
standard input.

## Community-maintained binaries[¶][24]

These binaries are maintained by the community—*thank you!*

Homebrew Arch Linux

Run in a terminal on macOS or Linux:

`brew install cargo-nextest
`

Run in a terminal:

`pacman -S cargo-nextest
`

## Using pre-built binaries in CI[¶][25]

Pre-built binaries can be used in continuous integration to speed up test runs.

### Using nextest in GitHub Actions[¶][26]

The easiest way to install nextest in GitHub Actions is to use the [Install Development Tools][27]
action maintained by [Taiki Endo][28].

To install the latest version of nextest, add this to your job after installing Rust and Cargo:

`- uses: taiki-e/install-action@nextest
`

[See this in practice with nextest's own CI.][29]

The action will download pre-built binaries from the URL above and add them to `.cargo/bin`.

To install a version series or specific version, use this instead:

`- uses: taiki-e/install-action@v2
  with:
    tool: nextest
    ## version (defaults to "latest") can be a series like 0.9:
    # tool: nextest@0.9
    ## version can also be a specific version like 0.9.11:
    # tool: nextest@0.9.11
`

ANSI color codes

GitHub Actions supports ANSI color codes. To get color support for nextest (and Cargo), add this to
your workflow:

`env:
  CARGO_TERM_COLOR: always
`

For a full list of environment variables supported by nextest, see [*Environment variables*][30].

### Other CI systems[¶][31]

Install pre-built binaries on other CI systems by downloading and extracting the respective
archives, using the commands above as a guide. See [*Release URLs*][32] for more about how to
specify nextest versions and platforms.

Documentation

If you've made it easy to install nextest on another CI system, please feel free to [submit a pull
request][33] with documentation.

## Release URLs[¶][34]

The latest nextest release is available at:

* [**get.nexte.st/latest/linux**][35] for Linux x86_64, including Windows Subsystem for Linux
  (WSL)^{[1][36]}
* [**get.nexte.st/latest/linux-musl**][37] for a statically linked Linux x86_64 binary, including
  Windows Subsystem for Linux (WSL)
* [**get.nexte.st/latest/linux-arm**][38] for Linux aarch64^{[1][39]}
* [**get.nexte.st/latest/mac**][40] for macOS, both x86_64 and Apple Silicon
* [**get.nexte.st/latest/windows**][41] for Windows x86_64
* [**get.nexte.st/latest/illumos**][42] for illumos x86_64
Other platforms

Nextest's CI isn't run on these platforms -- these binaries most likely work but aren't guaranteed
to do so.

* [**get.nexte.st/latest/windows-x86**][43] for Windows i686
* [**get.nexte.st/latest/freebsd**][44] for FreeBSD x86_64

These archives contain a single binary called `cargo-nextest` (`cargo-nextest.exe` on Windows). Add
this binary to a location on your PATH.

For a full specification of release URLs, see [*Release URLs*][45].

## Code signing policy[¶][46]

Nextest's Windows binaries are digitally signed. Free code signing is provided by SignPath.io, and
the certificate by SignPath Foundation.

* Committers and reviewers: [Members team][47]
* Approvers: [Owners][48]

This program will not transfer any information to other networked systems unless specifically
requested by the user or the person installing or operating it, e.g. via the [self-update
feature][49].

1. The standard Linux binaries target glibc, and have a minimum requirement of glibc 2.27 (Ubuntu
   18.04). [↩][50][↩][51]
November 17, 2025 May 27, 2024

[1]: #installing-pre-built-binaries
[2]: #downloading-and-installing-from-your-terminal
[3]: https://github.com/ryankurte/cargo-binstall
[4]: https://rustup.rs
[5]: ../../stability/
[6]: https://musl.libc.org/
[7]: https://rustup.rs
[8]: ../../stability/
[9]: https://rustup.rs
[10]: ../../stability/
[11]: https://rustup.rs
[12]: ../../stability/
[13]: https://rustup.rs
[14]: ../../stability/
[15]: https://rustup.rs
[16]: ../../stability/
[17]: https://rustup.rs
[18]: ../../stability/
[19]: https://rustup.rs
[20]: ../../stability/
[21]: https://rustup.rs
[22]: ../../stability/
[23]: https://www.illumos.org/issues/15228
[24]: #community-maintained-binaries
[25]: #using-pre-built-binaries-in-ci
[26]: #using-nextest-in-github-actions
[27]: https://github.com/marketplace/actions/install-development-tools
[28]: https://github.com/taiki-e
[29]: https://github.com/nextest-rs/nextest/blob/5b59a5c5d1a051ce651e5d632c93a849f97a9d4b/.github/wo
rkflows/ci.yml#L101-L102
[30]: ../../configuration/env-vars/
[31]: #other-ci-systems
[32]: ../release-urls/
[33]: https://github.com/nextest-rs/nextest/pulls
[34]: #release-urls
[35]: https://get.nexte.st/latest/linux
[36]: #fn:glibc
[37]: https://get.nexte.st/latest/linux
[38]: https://get.nexte.st/latest/linux-arm
[39]: #fn:glibc
[40]: https://get.nexte.st/latest/mac
[41]: https://get.nexte.st/latest/windows
[42]: https://get.nexte.st/latest/illumos
[43]: https://get.nexte.st/latest/windows-x86
[44]: https://get.nexte.st/latest/freebsd
[45]: ../release-urls/
[46]: #code-signing-policy
[47]: https://github.com/orgs/nextest-rs/teams/members
[48]: https://github.com/orgs/nextest-rs/people?query=role%3Aowner
[49]: ../updating/
[50]: #fnref:glibc
[51]: #fnref2:glibc
