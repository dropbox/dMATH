# [honggfuzz][1]

Security oriented software fuzzer. Supports evolutionary, feedback-driven fuzzing based on code
coverage (SW and HW based)

[View the Project on GitHub google/honggfuzz][2]

# Honggfuzz

## Description

A security oriented, feedback-driven, evolutionary, easy-to-use fuzzer with interesting analysis
options. See the [Usage document][3] for a primer on Honggfuzz use.

## Code

* Latest stable version: [2.6][4]
* [Changelog][5]

## Installation

`sudo apt-get install binutils-dev libunwind-dev libblocksruntime-dev clang
make
`

## Features

* It’s **multi-process** and **multi-threaded**: there’s no need to run multiple copies of your
  fuzzer, as honggfuzz can unlock potential of all your available CPU cores with a single running
  instance. The file corpus is automatically shared and improved between all fuzzed processes.
* It’s blazingly fast when the [persistent fuzzing mode][6] is used. A simple/empty
  *LLVMFuzzerTestOneInput* function can be tested with **up to 1mo iterations per second** on a
  relatively modern CPU (e.g. i7-6700K).
* Has a [solid track record][7] of uncovered security bugs: the **only** (to the date)
  **vulnerability in OpenSSL with the [critical][8] score mark** was discovered by honggfuzz. See
  the [Trophies][9] paragraph for the summary of findings to the date.
* Uses low-level interfaces to monitor processes (e.g. *ptrace* under Linux and NetBSD). As opposed
  to other fuzzers, it **will discover and report hijacked/ignored signals from crashes**
  (intercepted and potentially hidden by a fuzzed program).
* Easy-to-use, feed it a simple corpus directory (can even be empty for the [feedback-driven
  fuzzing][10]), and it will work its way up, expanding it by utilizing feedback-based coverage
  metrics.
* Supports several (more than any other coverage-based feedback-driven fuzzer) hardware-based (CPU:
  branch/instruction counting, **Intel BTS**, **Intel PT**) and software-based [feedback-driven
  fuzzing][11] modes. Also, see the new **[qemu mode][12]** for blackbox binary fuzzing.
* Works (at least) under GNU/Linux, FreeBSD, NetBSD, Mac OS X, Windows/CygWin and [Android][13].
* Supports the **persistent fuzzing mode** (long-lived process calling a fuzzed API repeatedly).
  More on that can be found [here][14].
* It comes with the **[examples][15] directory**, consisting of real world fuzz setups for
  widely-used software (e.g. Apache HTTPS, OpenSSL, libjpeg etc.).
* Provides a **[corpus minimization][16]** mode.


## Requirements

* **Linux** - The BFD library (libbfd-dev) and libunwind (libunwind-dev/libunwind8-dev), clang-5.0
  or higher for software-based coverage modes
* **FreeBSD** - gmake, clang-5.0 or newer
* **NetBSD** - gmake, clang, capstone, libBlocksRuntime
* **Android** - Android SDK/NDK. Also see [this detailed doc][17] on how to build and run it
* **Windows** - CygWin
* **Darwin/OS X** - Xcode 10.8+
* if **Clang/LLVM** is used to compile honggfuzz - link it with the BlocksRuntime Library
  (libblocksruntime-dev)

## Trophies

Honggfuzz has been used to find a few interesting security problems in major software packages; An
incomplete list:

* Dozens of security problems via the [OSS-Fuzz][18] project
* [Pre-auth remote crash in **OpenSSH**][19]
* **Apache HTTPD**
  
  * [Remote crash in **mod_http2** • CVE-2017-7659][20]
  * [Use-after-free in **mod_http2** • CVE-2017-9789][21]
  * [Memory leak in **mod_auth_digest** • CVE-2017-9788][22]
  * [Out of bound access • CVE-2018-1301][23]
  * [Write after free in HTTP/2 • CVE-2018-1302][24]
  * [Out of bound read • CVE-2018-1303][25]
* Various **SSL** libs
  
  * [Remote OOB read in **OpenSSL** • CVE-2015-1789][26]
  * [Remote Use-after-Free (potential RCE, rated as **critical**) in **OpenSSL** •
    CVE-2016-6309][27]
  * [Remote OOB write in **OpenSSL** • CVE-2016-7054][28]
  * [Remote OOB read in **OpenSSL** • CVE-2017-3731][29]
  * [Uninitialized mem use in **OpenSSL**][30]
  * [Crash in **LibreSSL**][31]
  * [Invalid free in **LibreSSL**][32]
  * [Uninitialized mem use in **BoringSSL**][33]
* [Adobe **Flash** memory corruption • CVE-2015-0316][34]
* [Multiple bugs in the **libtiff** library][35]
* [Multiple bugs in the **librsvg** library][36]
* [Multiple bugs in the **poppler** library][37]
* [Multiple exploitable bugs in **IDA-Pro**][38]
* [Remote DoS in **Crypto++** • CVE-2016-9939][39]
* Programming language interpreters
  
  * [**PHP/Python/Ruby**][40]
  * [PHP WDDX][41]
  * [PHP][42]
  * Perl: [#1][43], [#2][44], [#3][45]
* [Double-free in **LibXMP**][46]
* [Heap buffer overflow in SAPCAR • CVE-2017-8852][47]
* [Crashes in **libbass**][48]
* **FreeType 2**:
  
  * [CVE-2010-2497][49]
  * [CVE-2010-2498][50]
  * [CVE-2010-2499][51]
  * [CVE-2010-2500][52]
  * [CVE-2010-2519][53]
  * [CVE-2010-2520][54]
  * [CVE-2010-2527][55]
* Stack corruption issues in the Windows OpenType parser: [#1][56], [#2][57], [#3][58]
* [Infinite loop in **NGINX Unit**][59]
* A couple of problems in the [**MATLAB MAT File I/O Library**][60]: [#1][61], [#2][62], [#3][63],
  [#4][64], [#5][65]
* [**NASM**][66] [#1][67], [#2][68], [#3][69], [#4][70], [#5][71], [#6][72], [#7][73], [#8][74],
  [#9][75], [#10][76]
* **Samba** [tdbdump + tdbtool][77], [#2][78], [#3][79], [#4][80], [#5][81], [#6][82]
  [CVE-2019-14907][83] [CVE-2020-10745][84] [CVE-2021-20277][85] [LPRng_time][86]
* [Crash in **djvulibre**][87]
* [Multiple crashes in **VLC**][88]
* [Buffer overflow in **ClassiCube**][89]
* [Heap buffer-overflow (or UAF) in **MPV**][90]
* [Heap buffer-overflow in **picoc**][91]
* Crashes in **OpenCOBOL**: [#1][92], [#2][93]
* DoS in **ProFTPD**: [#1][94] • [#2][95]
* [Multiple security problems in ImageIO (iOS/MacOS)][96]
* [Memory corruption in **htmldoc**][97]
* [Memory corruption in **OpenDetex**][98]
* [Memory corruption in **Yabasic**][99]
* [Memory corruption in **Xfig**][100]
* [Memory corruption in **LibreOffice**][101]
* [Memory corruption in **ATasm**][102]
* [Memory corruption in **oocborrt**][103] • [CVE-2020-24753][104]
* [Memory corruption in **LibRaw**][105]
* [NULL-ptr deref in **peg-markdown**][106]
* [Uninitialized value in **MD4C**][107] • [CVE-2020-26148][108]
* [17 new bugs in **fwupd**][109]
* [Assertion in **libvips**][110]
* [Crash in
  **libocispec**)(https://github.com/containers/libocispec/commit/6079cd9490096cfb46752bd7491c712534
  18a02c)
* **Rust**:
  
  * panic() in regex [#1][111], [#2][112], [#3][113]
  * panic() in h2 [#1][114], [#2][115], [#3][116]
  * panic() in sleep-parser [#1][117]
  * panic() in lewton [#1][118]
  * panic()/DoS in Ethereum-Parity [#1][119]
  * crash() in Parts - a GPT partition manager [#1][120]
  * crashes in rust-bitcoin/rust-lightning [#1][121]
* … and more

## Projects utilizing or inspired-by Honggfuzz

* [**QuickFuzz** by CIFASIS][122]
* [**OSS-Fuzz**][123]
* [**Frog And Fuzz**][124]
* [**interpreters fuzzing**: by dyjakan][125]
* [**riufuzz**: honggfuzz with AFL-like UI][126]
* [**h2fuzz**: fuzzing Apache’s HTTP/2 implementation][127]
* [**honggfuzz-dharma**: honggfuzz with dharma grammar fuzzer][128]
* [**Owl**: a system for finding concurrency attacks][129]
* [**honggfuzz-docker-apps**][130]
* [**FFW**: Fuzzing For Worms][131]
* [**honggfuzz-rs**: fuzzing Rust with Honggfuzz][132]
* [**roughenough-fuzz**][133]
* [**Monkey**: a HTTP server][134]
* [**Killerbeez API**: a modular fuzzing framework][135]
* [**FuzzM**: a gray box model-based fuzzing framework][136]
* [**FuzzOS**: by Mozilla Security][137]
* [**Android**: by OHA][138]
* [**QDBI**: by Quarkslab][139]
* [**fuzzer-test-suite**: by Google][140]
* [**DeepState**: by Trail-of-Bits][141]
* [**Quiche-HTTP/3**: by Cloudflare][142]
* [**Bolero**: fuzz and property testing framework][143]
* [**pwnmachine**: a vagrantfile for exploit development on Linux][144]
* [**Quick700**: analyzing effectiveness of fuzzers on web browsers and web servers][145]
* [**python-hfuzz**: gluing honggfuzz and python3][146]
* [**go-hfuzz**: gluing honggfuzz and go][147]
* [**Magma**: a ground-truth fuzzing benchmark][148]
* [**arbitrary-model-tests**: a procedural macro for testing stateful models][149]
* [**Clusterfuzz**: the fuzzing engine behind OSS-fuzz/Chrome-fuzzing][150]
* [**Apache HTTP Server**][151]
* [**centos-fuzz**][152]
* [**FLUFFI**: Fully Localized Utility For Fuzzing Instantaneously by Siemens][153]
* [**Fluent Bit**: a fast log processor and forwarder for Linux][154]
* [**Samba**: a SMB server][155]
* [**universal-fuzzing-docker**: by nnamon][156]
* [**Canokey Core**: core implementations of an open-source secure key][157]
* [**uberfuzz2**: a cooperative fuzzing framework][158]
* [**TiKV**: a distributed transactional key-value database][159]
* [**fuzz-monitor**][160]
* [**libmutator**: a C library intended to generate random test cases by mutating legitimate test
  cases][161]
* [**StatZone**: a DNS zone file analyzer][162]
* [**shub-fuzz/honggfuzz**: singularity image for honggfuzz][163]
* [**Code Intelligence**: fuzzing-as-a-service][164]
* [**SpecFuzz**: fuzzing for Spectre vulnerabilities][165]
* [**rcc**: a Rust C compiler][166]
* [**EIP1962Fuzzing**: Fuzzy testing of various EIP1962 implementations][167]
* [**wasm-fuzz**: Fuzzing of wasmer][168], [blog post][169]
* [**propfuzz**: Rust tools to combine coverage-guided fuzzing with property-based testing - from
  Facebook][170]
* [**Bitcoin Core**: fuzzing][171]
* [**ESP32-Fuzzing-Framework**: A Fuzzing Framework for ESP32 applications][172]
* [**Fuzzbench**: Fuzzer Benchmarking As a Service][173]
* [**rumpsyscallfuzz**: NetBSD Rump Kernel fuzzing][174]
* [**libnbd**: fuzzing libnbd with honggfuzz][175]
* [**EnsmallenGraph**: Rust library to run node2vec-like weighted random walks on very big
  graphs][176]
* [**Oasis Core**][177]
* [**bp7-rs**: Rust implementation of dtn bundle protocol 7][178]
* [**WHATWG**: URL C++ library][179]
* [**Xaya Core / Chimera**: A decentralized open source information registration and transfer
  system][180]
* [**OpenWRT**: A Linux operating system targeting embedded devices][181]
* [**RcppDeepStateTools**: A Linux-specific R package, with R functions for running the DeepState
  test harness][182]
* [**Materialize**: A streaming database for real-time applications][183]
* [**Rust-Bitcoin**][184]
* [**Substrate**: A next-generation framework for blockchain innovation][185]
* [**Solana**: A fast, secure, and censorship resistant blockchain][186]
* [**fwupd**: A project that aims to make updating firmware on Linux automatic, safe and
  reliable][187]
* [**polkadot**: Implementation of a https://polkadot.network node in Rust based on the Substrate
  framework][188]
* [**systemd**: is tested by honggfuzz][189]
* [**freetype**: is tested by honggfuzz][190]
* [**ghostscript**: is tested by honggfuzz][191]
* [**Fuzzme**: fuzzing templates for programming languages and fuzzers][192]
* [**P0**: Fuzzing ImageIO][193]
  
  * [**TrapFuzz**: by P0][194]
* [**Rust’s fuzztest**][195]
  
  * [*and multiple Rust projects*][196]

## Contact

* User mailing list: [honggfuzz@googlegroups.com][197], sign up with [this link][198].

**This is NOT an official Google product**

This project is maintained by [google][199]

Hosted on GitHub Pages — Theme by [orderedlist][200]

[1]: https://honggfuzz.dev/
[2]: https://github.com/google/honggfuzz
[3]: https://github.com/google/honggfuzz/blob/master/docs/USAGE.md
[4]: https://github.com/google/honggfuzz/releases
[5]: https://github.com/google/honggfuzz/blob/master/CHANGELOG
[6]: https://github.com/google/honggfuzz/blob/master/docs/PersistentFuzzing.md
[7]: #trophies
[8]: https://www.openssl.org/news/secadv/20160926.txt
[9]: #trophies
[10]: https://github.com/google/honggfuzz/blob/master/docs/FeedbackDrivenFuzzing.md
[11]: https://github.com/google/honggfuzz/blob/master/docs/FeedbackDrivenFuzzing.md
[12]: https://github.com/google/honggfuzz/tree/master/qemu_mode
[13]: https://github.com/google/honggfuzz/blob/master/docs/Android.md
[14]: https://github.com/google/honggfuzz/blob/master/docs/PersistentFuzzing.md
[15]: https://github.com/google/honggfuzz/tree/master/examples
[16]: https://github.com/google/honggfuzz/blob/master/docs/USAGE.md#corpus-minimization--m
[17]: https://github.com/google/honggfuzz/blob/master/docs/Android.md
[18]: https://bugs.chromium.org/p/oss-fuzz/issues/list?q=honggfuzz&can=1
[19]: https://anongit.mindrot.org/openssh.git/commit/?id=28652bca29046f62c7045e933e6b931de1d16737
[20]: http://seclists.org/oss-sec/2017/q2/504
[21]: http://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-9789
[22]: http://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-9788
[23]: http://seclists.org/oss-sec/2018/q1/265
[24]: http://seclists.org/oss-sec/2018/q1/268
[25]: http://seclists.org/oss-sec/2018/q1/266
[26]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2015-1789
[27]: https://www.openssl.org/news/secadv/20160926.txt
[28]: https://www.openssl.org/news/secadv/20161110.txt
[29]: https://www.openssl.org/news/secadv/20170126.txt
[30]: https://github.com/openssl/openssl/commit/bd5d27c1c6d3f83464ddf5124f18a2cac2cbb37f
[31]: https://github.com/openbsd/src/commit/c80d04452814d5b0e397817ce4ed34edb4eb520d
[32]: https://ftp.openbsd.org/pub/OpenBSD/LibreSSL/libressl-2.6.2-relnotes.txt
[33]: https://github.com/boringssl/boringssl/commit/7dccc71e08105b100c3acd56fa5f6fc1ba9b71d3
[34]: http://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2015-0316
[35]: http://bugzilla.maptools.org/buglist.cgi?query_format=advanced;emailreporter1=1;email1=robert@
swiecki.net;product=libtiff;emailtype1=substring
[36]: https://bugzilla.gnome.org/buglist.cgi?query_format=advanced;emailreporter1=1;email1=robert%40
swiecki.net;product=librsvg;emailtype1=substring
[37]: http://lists.freedesktop.org/archives/poppler/2010-November/006726.html
[38]: https://www.hex-rays.com/bugbounty.shtml
[39]: http://www.openwall.com/lists/oss-security/2016/12/12/7
[40]: https://github.com/dyjakan/interpreter-bugs
[41]: https://bugs.php.net/bug.php?id=74145
[42]: https://bugs.php.net/bug.php?id=74194
[43]: https://www.nntp.perl.org/group/perl.perl5.porters/2018/03/msg250072.html
[44]: https://github.com/Perl/perl5/issues/16468
[45]: https://github.com/Perl/perl5/issues/16015
[46]: https://github.com/cmatsuoka/libxmp/commit/bd1eb5cfcd802820073504c234c3f735e96c3355
[47]: https://www.coresecurity.com/blog/sapcar-heap-buffer-overflow-crash-exploit
[48]: http://seclists.org/oss-sec/2017/q4/185
[49]: https://bugzilla.redhat.com/show_bug.cgi?id=CVE-2010-2497
[50]: https://bugzilla.redhat.com/show_bug.cgi?id=CVE-2010-2498
[51]: https://bugzilla.redhat.com/show_bug.cgi?id=CVE-2010-2499
[52]: https://bugzilla.redhat.com/show_bug.cgi?id=CVE-2010-2500
[53]: https://bugzilla.redhat.com/show_bug.cgi?id=CVE-2010-2519
[54]: https://bugzilla.redhat.com/show_bug.cgi?id=CVE-2010-2520
[55]: https://bugzilla.redhat.com/show_bug.cgi?id=CVE-2010-2527
[56]: https://github.com/xinali/AfdkoFuzz/blob/4eadcb19eacb2fb73e4b0f0b34f382a9331bb3b4/CrashesAnaly
sis/CrashesAnalysis_3/README.md
[57]: https://github.com/xinali/AfdkoFuzz/blob/master/CVE-2019-1117/README.md
[58]: https://github.com/xinali/AfdkoFuzz/tree/f6d6562dd19403cc5a1f8cef603ee69425b68b20/CVE-2019-111
8
[59]: https://github.com/nginx/unit/commit/477e8177b70acb694759e62d830b8a311a736324
[60]: https://sourceforge.net/projects/matio
[61]: https://github.com/tbeu/matio/commit/406438f497931f45fb3edf6de17d3a59a922c257
[62]: https://github.com/tbeu/matio/commit/406438f497931f45fb3edf6de17d3a59a922c257
[63]: https://github.com/tbeu/matio/commit/a55b9c2c01582b712d5a643699a13b5c41687db1
[64]: https://github.com/tbeu/matio/commit/3e6283f37652e29e457ab9467f7738a562594b6b
[65]: https://github.com/tbeu/matio/commit/783ee496a6914df68e77e6019054ad91e8ed6420
[66]: https://github.com/netwide-assembler/nasm
[67]: https://bugzilla.nasm.us/show_bug.cgi?id=3392501
[68]: https://bugzilla.nasm.us/show_bug.cgi?id=3392750
[69]: https://bugzilla.nasm.us/show_bug.cgi?id=3392751
[70]: https://bugzilla.nasm.us/show_bug.cgi?id=3392760
[71]: https://bugzilla.nasm.us/show_bug.cgi?id=3392761
[72]: https://bugzilla.nasm.us/show_bug.cgi?id=3392762
[73]: https://bugzilla.nasm.us/show_bug.cgi?id=3392792
[74]: https://bugzilla.nasm.us/show_bug.cgi?id=3392793
[75]: https://bugzilla.nasm.us/show_bug.cgi?id=3392795
[76]: https://bugzilla.nasm.us/show_bug.cgi?id=3392796
[77]: http://seclists.org/oss-sec/2018/q2/206
[78]: https://github.com/samba-team/samba/commit/183da1f9fda6f58cdff5cefad133a86462d5942a
[79]: https://github.com/samba-team/samba/commit/33e9021cbee4c17ee2f11d02b99902a742d77293
[80]: https://github.com/samba-team/samba/commit/ac1be895d2501dc79dcff2c1e03549fe5b5a930c
[81]: https://github.com/samba-team/samba/commit/b1eda993b658590ebb0a8225e448ce399946ed83
[82]: https://github.com/samba-team/samba/commit/f7f92803f600f8d302cdbb668c42ca8b186a797f
[83]: https://www.samba.org/samba/security/CVE-2019-14907.html
[84]: https://www.samba.org/samba/security/CVE-2020-10745.html
[85]: https://www.samba.org/samba/security/CVE-2021-20277.html
[86]: https://github.com/smokey57/samba/commit/fc267567a072c9483bbcc5cc18e150244bc5376b
[87]: https://github.com/barak/djvulibre/commit/89d71b01d606e57ecec2c2930c145bb20ba5bbe3
[88]: https://www.pentestpartners.com/security-blog/double-free-rce-in-vlc-a-honggfuzz-how-to/
[89]: https://github.com/UnknownShadow200/ClassiCube/issues/591
[90]: https://github.com/mpv-player/mpv/issues/6808
[91]: https://gitlab.com/zsaleeba/picoc/issues/44
[92]: https://sourceforge.net/p/open-cobol/bugs/586/
[93]: https://sourceforge.net/p/open-cobol/bugs/587/
[94]: https://twitter.com/SecReLabs/status/1186548245553483783
[95]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-18217
[96]: https://googleprojectzero.blogspot.com/2020/04/fuzzing-imageio.html
[97]: https://github.com/michaelrsweet/htmldoc/issues/370
[98]: https://github.com/pkubowicz/opendetex/issues/60
[99]: https://github.com/marcIhm/yabasic/issues/36
[100]: https://sourceforge.net/p/mcj/tickets/67/
[101]: https://github.com/LibreOffice/core/commit/0754e581b0d8569dd08cf26f88678754f249face
[102]: https://sourceforge.net/p/atasm/bugs/8/
[103]: https://warcollar.com/cve-2020-24753.html
[104]: https://nvd.nist.gov/vuln/detail/CVE-2020-24753
[105]: https://github.com/LibRaw/LibRaw/issues/309
[106]: https://github.com/jgm/peg-markdown/issues/43
[107]: https://github.com/mity/md4c/issues/130
[108]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-26148
[109]: https://github.com/google/oss-fuzz/pull/4823#issue-537143670
[110]: https://github.com/libvips/libvips/issues/1890
[111]: https://github.com/rust-lang/regex/issues/464
[112]: https://github.com/rust-lang/regex/issues/465
[113]: https://github.com/rust-lang/regex/issues/465#issuecomment-381412816
[114]: https://github.com/carllerche/h2/pull/260
[115]: https://github.com/carllerche/h2/pull/261
[116]: https://github.com/carllerche/h2/pull/262
[117]: https://github.com/datrs/sleep-parser/issues/3
[118]: https://github.com/RustAudio/lewton/issues/27
[119]: https://srlabs.de/bites/ethereum_dos/
[120]: https://github.com/DianaNites/parts/commit/d8ab05d48d87814f362e94f01c93d9eeb4f4abf4
[121]: https://github.com/rust-bitcoin/rust-lightning/commit/a9aa3c37fe182dd266e0faebc788e0c9ee72478
3
[122]: http://quickfuzz.org
[123]: https://github.com/google/oss-fuzz
[124]: https://github.com/warsang/FrogAndFuzz/tree/develop
[125]: https://github.com/dyjakan/interpreter-bugs
[126]: https://github.com/riusksk/riufuzz
[127]: https://github.com/icing/h2fuzz
[128]: https://github.com/Sbouber/honggfuzz-dharma
[129]: https://github.com/hku-systems/owl
[130]: https://github.com/skysider/honggfuzz_docker_apps
[131]: https://github.com/dobin/ffw
[132]: https://docs.rs/honggfuzz/
[133]: https://github.com/int08h/roughenough-fuzz
[134]: https://github.com/monkey/monkey/blob/master/FUZZ.md
[135]: https://github.com/grimm-co/killerbeez
[136]: https://github.com/collins-research/FuzzM
[137]: https://github.com/MozillaSecurity/fuzzos
[138]: https://android.googlesource.com/platform/external/honggfuzz
[139]: https://project.inria.fr/FranceJapanICST/files/2019/04/19-Kyoto-Fuzzing_Binaries_using_Dynami
c_Instrumentation.pdf
[140]: https://github.com/google/fuzzer-test-suite
[141]: https://github.com/trailofbits/deepstate
[142]: https://github.com/cloudflare/quiche/pull/179
[143]: https://github.com/camshaft/bolero
[144]: https://github.com/kapaw/pwnmachine/commit/9cbfc6f1f9547ed2d2a5d296f6d6cd8fac0bb7e1
[145]: https://github.com/Quick700/Quick700
[146]: https://github.com/thebabush/python-hfuzz
[147]: https://github.com/thebabush/go-hfuzz
[148]: https://github.com/HexHive/magma
[149]: https://github.com/jakubadamw/arbitrary-model-tests
[150]: https://github.com/google/clusterfuzz/issues/1128
[151]: https://github.com/apache/httpd/commit/d7328a07d7d293deb5ce62a60c2ce6029104ebad
[152]: https://github.com/truelq/centos-fuzz
[153]: https://github.com/siemens/fluffi
[154]: https://github.com/fluent/fluent-bit/search?q=honggfuzz&unscoped_q=honggfuzz
[155]: https://github.com/samba-team/samba/blob/2a90202052558c945e02675d1331e65aeb15f9fa/lib/fuzzing
/README.md
[156]: https://github.com/nnamon/universal-fuzzing-docker
[157]: https://github.com/canokeys/canokey-core/search?q=honggfuzz&unscoped_q=honggfuzz
[158]: https://github.com/acidghost/uberfuzz2
[159]: https://github.com/tikv/tikv/tree/99a922564face31bdb59b5b38962339f79e0015c/fuzz
[160]: https://github.com/acidghost/fuzz-monitor/search?q=honggfuzz&unscoped_q=honggfuzz
[161]: https://github.com/denandz/libmutator
[162]: https://github.com/fcambus/statzone
[163]: https://github.com/shub-fuzz/honggfuzz
[164]: https://www.code-intelligence.com/technology.html
[165]: https://github.com/OleksiiOleksenko/SpecFuzz
[166]: https://github.com/jyn514/rcc#testing
[167]: https://github.com/matter-labs/eip1962_fuzzing
[168]: https://github.com/wasmerio/wasm-fuzz/blob/master/honggfuzz.md
[169]: https://medium.com/wasmer/fuzz-testing-in-webassembly-vms-3a301f982e5a
[170]: https://github.com/facebookincubator/propfuzz
[171]: https://github.com/Nampu898/btc-2/blob/2af56d6d5c387c3208d3d5aae8d428a3d610446f/doc/fuzzing.m
d#fuzzing-bitcoin-core-using-honggfuzz
[172]: https://github.com/MaxCamillo/esp32-fuzzing-framework/tree/5130a3c7bf9796fdeb44346eec3dcdc7e5
07a62b
[173]: https://www.fuzzbench.com/
[174]: https://github.com/adityavardhanpadala/rumpsyscallfuzz
[175]: https://github.com/libguestfs/libnbd/commit/329c5235f81ab0d1849946bab5e5c4119b35e140
[176]: https://github.com/LucaCappelletti94/ensmallen_graph/
[177]: https://github.com/oasisprotocol/oasis-core/
[178]: https://github.com/dtn7/bp7-rs
[179]: https://github.com/rmisev/url_whatwg/commit/0bb2821ccab170c7b12b45524a2196eb7bf35e0b
[180]: https://github.com/xaya/xaya/commit/b337bd7bc0873ace317ad8e1ebbd3842da3f81d5
[181]: https://github.com/ynezz/openwrt-ci/commit/70956d056b1d041c28b76e9e06574d511b428f68
[182]: https://github.com/akhikolla/RcppDeepStateTools/commit/0b85b0b8b2ab357a0840f45957e2cb285d98d4
30
[183]: https://github.com/MaterializeInc/materialize/pull/5519/commits/5eb09adb687c4980fc899582cefaa
5e43d6e8ce7
[184]: https://github.com/rust-bitcoin/rust-lightning/pull/782
[185]: https://github.com/rakanalh/substrate/pull/5
[186]: https://github.com/solana-labs/solana/issues/14707
[187]: https://github.com/fwupd/fwupd/pull/2666
[188]: https://github.com/paritytech/polkadot/pull/2021/commits/b731cfa34e330489ecd832b058e82ce2b88f
75f5
[189]: https://github.com/systemd/systemd/commit/d2c3f14fed67e7246adfdeeb5957c0d0497d7dc7
[190]: https://github.com/freetype/freetype2-testing/commit/e401ce29d7bfe37cfd0085c244e213c913221b5f
[191]: https://github.com/google/oss-fuzz/commit/365df31265438684a50c500e7d9355744fd7965d
[192]: https://github.com/ForAllSecure/fuzzme
[193]: https://googleprojectzero.blogspot.com/2020/04/fuzzing-imageio.html
[194]: https://github.com/googleprojectzero/p0tools/tree/master/TrapFuzz
[195]: https://docs.rs/crate/fuzztest
[196]: https://github.com/search?q=%22extern+crate+honggfuzz%22&type=Code
[197]: mailto:honggfuzz@googlegroups.com
[198]: https://groups.google.com/forum/#!forum/honggfuzz
[199]: https://github.com/google
[200]: https://github.com/orderedlist
