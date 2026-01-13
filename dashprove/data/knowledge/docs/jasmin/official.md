# Jasmin

[[pipeline status]][1] [[project chat]][2] [[project chat]][3]

## About

Jasmin denotes both a language and a compiler designed for writing high-assurance and high-speed
cryptography. Jasmin implementations aim at being efficient, safe, correct, and secure.

Reference documentation of the language and compiler are on [readthedocs][4].

## Installation

For a complete installation guide covering several usecases, please read our [documentation][5].

If you just want to install the Jasmin tools, here is a TL;DR:

* with APT (debian, ubuntu), a package is available in a dedicated repository, see the
  [documentation][6]
* with nix
  
  `nix-env -iA nixpkgs.jasmin-compiler
  `
* with opam
  
  `opam install jasmin
  `

## Getting support

The [Formosa-Crypto Zulip Chat][7] is meant for anybody interested in high-assurance cryptography
using EasyCrypt, Jasmin, and related tools.

## License

Jasmin is free software. All files in this distribution are, unless specified otherwise, licensed
under the [MIT license][8]. The documentation (under `docs/`) is licensed separately from the
compiler, under the [CC-BY 4.0][9].

[1]: https://gitlab.com/jasmin-lang/jasmin/-/commits/main
[2]: https://jasmin-lang.readthedocs.org
[3]: https://formosa-crypto.zulipchat.com
[4]: https://jasmin-lang.readthedocs.io
[5]: https://jasmin-lang.readthedocs.io/en/stable/misc/installation_guide.html
[6]: https://jasmin-lang.readthedocs.io/en/stable/misc/installation_guide.html#on-debian-and-related
-linux-distributions
[7]: https://formosa-crypto.zulipchat.com/
[8]: /jasmin-lang/jasmin/blob/main/LICENSE
[9]: /jasmin-lang/jasmin/blob/main/docs/LICENSE
