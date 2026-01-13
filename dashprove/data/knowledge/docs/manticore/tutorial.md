[ trailofbits ][1] / ** [manticore][2] ** Public

* [ Notifications ][3] You must be signed in to change notification settings
* [ Fork 490 ][4]
* [ Star 3.8k ][5]

* [ Code ][6]
* [ Issues 262 ][7]
* [ Pull requests 27 ][8]
* [ Discussions ][9]
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
* [ Discussions ][19]
* [ Actions ][20]
* [ Projects ][21]
* [ Wiki ][22]
* [ Security ][23]
* [ Insights ][24]

# Home

[Jump to bottom][25] [ Edit ][26] [ New page ][27]
ConstantinHvber edited this page Feb 8, 2022 · [81 revisions][28]

Welcome to the manticore wiki!

## Stable Version (0.3.6)

[Manticore for Windows][29]

[Manticore for Mac][30]

[Manticore for Linux][31]

[[Build Status]][32] [[Coverage Status]][33] [[PyPI Version]][34] [[Slack Status]][35]
[[Documentation Status]][36] [[Example Status]][37] [[LGTM Total Alerts]][38]

## Documentation

* [Tutorial][39]
* [HTML API docs][40]
* [Join #manticore on our Slack][41]

## Examples

Explore the [examples][42] directory to find sample binaries and scripts that demonstrate the API.

We recommend starting with the following examples:

* [multiple-styles writeup][43]: demonstrates binary instrumentation and symbolic execution
* [introduce_symbolic_bytes.py][44]: demonstrates taint analysis
* [Building an exploit][45]: demonstrates crash analysis

More complex examples are also available in the [manticore-examples][46] repository

## Bounties

We're happy to offer bounties of $50, $100, or $200 for contributions to Manticore. [Mugs][47] and
[stickers][48] are also available.

[Contact us][49] for a bounty payout if you:

* Publish a challenge writeup. We'll add you to the list of references on this wiki.
* Implement a new [syscall][50] or [instruction][51]. Help us get coverage of more complex binaries.
* Fix any bug! Try looking through the [easy][52] and [help wanted][53] labels.

## FAQ

### How does Manticore compare to angr?

Manticore and Angr are both Python-based symbolic execution engines that provide an API for binary
analysis. Broadly speaking, Manticore is simpler. It has a smaller codebase, fewer dependencies and
features, and (in our opinion) an easier learning curve. In exchange, Manticore lacks angr's
advanced features like CFG recovery, ROP chain building, and binary patching. Manticore does not use
any intermediate representation, and overall emphasizes staying close to machine abstractions. Angr,
by contrast, raises native instructions to VEX IR, allowing it to support a wide variety of native
architectures. Finally, Manticore supports exotic architectures such as the Ethereum Virtual Machine
(EVM) and WebAssembly (WASM).

### Was Manticore part of the Trail of Bits CRS?

Not exactly. The [Trail of Bits CRS][54] used [FrankenPSE][55] to provide its binary symbolic
execution capabilities. FrankenPSE and Manticore share the same heritage: [PySymEmu][56] (2013). The
difference between the two stems from their respective use-cases.

Manticore is designed so an expert user can guide it, and therefore supports flexible APIs that help
its users achieve specific goals. Manticore also supports more architectures and binary file
formats.

FrankenPSE was designed to tightly integrate with the Trail of Bits CRS. This includes sharing the
same program snapshot representation as the [GRR fuzzer][57]. FrankenPSE is also x86-only and uses
[microx][58], a lightweight, single-instruction x86 instruction JIT executor.

## Troubleshooting

### On manticore version 0.2.2 EVM contracts terminate with INVALID/OOG with not apparent reason

This happens because of a known bug in that specific manticore version. The gas limits are too small
by default and then most complex transactions will just end with an OutOfGas exception. Even complex
constructors could end like this failing to make the contract account. Fix, just change the gas
defaults:

`import manticore.ethereum
manticore.ethereum.ManticoreEVM.create_contract.__defaults__=(0, None, None, None, 0xffffffffffff)
manticore.ethereum.ManticoreEVM.transaction.__defaults__=(0xffffffffffff,)
`

### "ImportError: ERROR: fail to load the dynamic library."

You ran Manticore and it errored on something like this:

`  File "/root/.virtualenvs/manticore/local/lib/python2.7/site-packages/manticore/core/cpu/abstractc
pu.py", line 1, in <module>
    from capstone import *
  File "/root/.virtualenvs/manticore/local/lib/python2.7/site-packages/capstone/__init__.py", line 2
30, in <module>
    raise ImportError("ERROR: fail to load the dynamic library.")
ImportError: ERROR: fail to load the dynamic library.
`

This is a [known issue][59] in capstone. Try reinstalling capstone with the --no-binary flag.

### I'm seeing "Invalid memory access" messages when I run Manticore on native binaries. I don't
### think these are correct. Is this a Manticore bug?

Maybe, but it might also be a bug in our disassembler dependency, [Capstone][60]. One way to check
is to try using the `--process-dependency-links` pip flag when installing Manticore. This will
install the development branch of Capstone, which may contain useful bug fixes and potentially
resolve the issue.

### Manticore was installed successfully, the API is accessible via py scripts, but the commandline
### `manticore` is not available

Your `$PATH` can be set up incorrectly. Try running manticore via `python -m manticore`.

It might be that you installed manticore with `sudo` (e.g. instead of in a Python virtual
environment) and so you don't have permission to run the manticore script.

If you are running our docker image, it might be that you overwrote the `/home/manticore` path (e.g.
by `-v some_dir_on_host:/home/manticore`) which contains the manticore script in `~/.local`
directory.

A deeper fix involves adding the correct directory to your `PATH` environment variable. On Linux you
can find manticore script by using `locate`:

`apt install locate
updatedb  # may require sudo, updates locate cache
locate bin/manticore
`

The last command will find all paths that contains `bin/manticore` and one of them should be your
script. This way you can investigate whether:

* you lack permissions
* the script is there or is missing from the system
* you are not in a proper Python virtual environment

## Citation

If you would like to cite Manticore, you can use this bibtex.

`@misc{trailofbits-manticore,
  title  = "Manticore: Symbolic Execution for Humans",
  author = "Trail of Bits",
  howpublished = {\url{https://github.com/trailofbits/manticore}}
}
`
[ Add a custom footer ][61]

## Toggle table of contents Pages 20

* Loading [
  Home
  ][62]
  
  * [Stable Version (0.3.6)][63]
  * [Documentation][64]
  * [Examples][65]
  * [Bounties][66]
  * [FAQ][67]
  * [How does Manticore compare to angr?][68]
  * [Was Manticore part of the Trail of Bits CRS?][69]
  * [Troubleshooting][70]
  * [On manticore version 0.2.2 EVM contracts terminate with INVALID/OOG with not apparent
    reason][71]
  * ["ImportError: ERROR: fail to load the dynamic library."][72]
  * [I'm seeing "Invalid memory access" messages when I run Manticore on native binaries. I don't
    think these are correct. Is this a Manticore bug?][73]
  * [Manticore was installed successfully, the API is accessible via py scripts, but the commandline
    manticore is not available][74]
  * [Citation][75]
* Loading [
  Analysis Design Choices
  ][76]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][77].
* Loading [
  Analysis Tips
  ][78]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][79].
* Loading [
  Docker notes
  ][80]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][81].
* Loading [
  Ethereum Detectors
  ][82]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][83].
* Loading [
  Getting Started EVM
  ][84]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][85].
* Loading [
  Getting Started Native
  ][86]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][87].
* Loading [
  Hacking on Manticore
  ][88]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][89].
* Loading [
  Ideas
  ][90]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][91].
* Loading [
  Plugins
  ][92]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][93].
* Loading [
  Python2 to 3 Compatibility Pitfalls
  ][94]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][95].
* Loading [
  Redis
  ][96]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][97].
* Loading [
  Releasing Manticore
  ][98]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][99].
* Loading [
  Tutorial
  ][100]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][101].
* Loading [
  Tutorial: Adding Constraints
  ][102]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][103].
* Loading [
  Tutorial: Exercise
  ][104]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][105].
* Loading [
  Tutorial: Getting Throwing Path
  ][106]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][107].
* Loading [
  Tutorial: Running under Manticore
  ][108]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][109].
* Loading [
  What's in the workspace?
  ][110]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][111].
* Loading [
  Wiki Page by Swapnil_Kothawade for POC
  ][112]
  
  ### Uh oh!
  
  
  There was an error while loading. [Please reload this page][113].
* Show 5 more pages…
[ ][114]

* [Home][115]
* [Getting Started Native][116]
* [Getting Started Evm][117]
* [Plugins][118]
* [Tutorial][119]
  
  * [Running under Manticore][120]
  * [How to get throwing path][121]
  * [How to add constraints][122]
  * [Exercise][123]
* [The Manticore Workspace][124]
* [Analysis Design Choices][125]
* [Analysis Tips][126]
* [Docker notes][127]
* [Ethereum Detectors][128]
* [Hacking on Manticore][129]
* [Ideas][130]
* [Releasing Manticore][131]

### Clone this wiki locally

[1]: /trailofbits
[2]: /trailofbits/manticore
[3]: /login?return_to=%2Ftrailofbits%2Fmanticore
[4]: /login?return_to=%2Ftrailofbits%2Fmanticore
[5]: /login?return_to=%2Ftrailofbits%2Fmanticore
[6]: /trailofbits/manticore
[7]: /trailofbits/manticore/issues
[8]: /trailofbits/manticore/pulls
[9]: /trailofbits/manticore/discussions
[10]: /trailofbits/manticore/actions
[11]: /trailofbits/manticore/projects
[12]: /trailofbits/manticore/wiki
[13]: /trailofbits/manticore/security
[14]: /trailofbits/manticore/security
[15]: /trailofbits/manticore/pulse
[16]: /trailofbits/manticore
[17]: /trailofbits/manticore/issues
[18]: /trailofbits/manticore/pulls
[19]: /trailofbits/manticore/discussions
[20]: /trailofbits/manticore/actions
[21]: /trailofbits/manticore/projects
[22]: /trailofbits/manticore/wiki
[23]: /trailofbits/manticore/security
[24]: /trailofbits/manticore/pulse
[25]: #wiki-pages-box
[26]: /trailofbits/manticore/wiki/Home/_edit
[27]: /trailofbits/manticore/wiki/_new
[28]: /trailofbits/manticore/wiki/Home/_history
[29]: https://linkify.me/Ve5TbqN
[30]: https://github.com/trailofbits/manticore/releases
[31]: https://github.com/trailofbits/manticore/releases
[32]: https://github.com/trailofbits/manticore/actions?query=workflow%3ACI
[33]: https://coveralls.io/github/trailofbits/manticore
[34]: https://badge.fury.io/py/manticore
[35]: https://empireslacking.herokuapp.com
[36]: http://manticore.readthedocs.io/en/latest/?badge=latest
[37]: https://github.com/trailofbits/manticore-examples/actions?query=workflow%3ACI
[38]: https://lgtm.com/projects/g/trailofbits/manticore/alerts/
[39]: wiki/Tutorial
[40]: https://manticore.readthedocs.io/en/latest/
[41]: https://empireslacking.herokuapp.com
[42]: https://github.com/trailofbits/manticore/tree/master/examples
[43]: https://gist.github.com/ehennenfent/a5ad9746615d1490c618a88b98769c10
[44]: https://github.com/trailofbits/manticore/blob/master/examples/script/introduce_symbolic_bytes.
py
[45]: https://github.com/trailofbits/manticore-examples/tree/master/test_exploit_generation_example
[46]: https://github.com/trailofbits/manticore-examples
[47]: https://twitter.com/trailofbits/status/849988694526365698
[48]: https://twitter.com/trailofbits/status/270978150081650688
[49]: https://empireslacking.herokuapp.com
[50]: wiki/Hacking-on-Manticore#adding-a-syscall
[51]: wiki/Hacking-on-Manticore#adding-an-instruction
[52]: https://github.com/trailofbits/manticore/labels/easy
[53]: https://github.com/trailofbits/manticore/labels/help%20wanted
[54]: https://blog.trailofbits.com/2015/07/15/how-we-fared-in-the-cyber-grand-challenge/
[55]: https://blog.trailofbits.com/2016/08/02/engineering-solutions-to-hard-program-analysis-problem
s/
[56]: https://github.com/feliam/pysymemu
[57]: https://github.com/trailofbits/grr
[58]: https://github.com/trailofbits/microx
[59]: https://github.com/aquynh/capstone/issues/445
[60]: https://github.com/aquynh/capstone/
[61]: /trailofbits/manticore/wiki/_new?wiki%5Bname%5D=_Footer
[62]: /trailofbits/manticore/wiki
[63]: /trailofbits/manticore/wiki#stable-version-036
[64]: /trailofbits/manticore/wiki#documentation
[65]: /trailofbits/manticore/wiki#examples
[66]: /trailofbits/manticore/wiki#bounties
[67]: /trailofbits/manticore/wiki#faq
[68]: /trailofbits/manticore/wiki#how-does-manticore-compare-to-angr
[69]: /trailofbits/manticore/wiki#was-manticore-part-of-the-trail-of-bits-crs
[70]: /trailofbits/manticore/wiki#troubleshooting
[71]: /trailofbits/manticore/wiki#on-manticore-version-022-evm-contracts-terminate-with-invalidoog-w
ith-not-apparent-reason
[72]: /trailofbits/manticore/wiki#importerror-error-fail-to-load-the-dynamic-library
[73]: /trailofbits/manticore/wiki#im-seeing-invalid-memory-access-messages-when-i-run-manticore-on-n
ative-binaries-i-dont-think-these-are-correct-is-this-a-manticore-bug
[74]: /trailofbits/manticore/wiki#manticore-was-installed-successfully-the-api-is-accessible-via-py-
scripts-but-the-commandline-manticore-is-not-available
[75]: /trailofbits/manticore/wiki#citation
[76]: /trailofbits/manticore/wiki/Analysis-Design-Choices
[77]: 
[78]: /trailofbits/manticore/wiki/Analysis-Tips
[79]: 
[80]: /trailofbits/manticore/wiki/Docker-notes
[81]: 
[82]: /trailofbits/manticore/wiki/Ethereum-Detectors
[83]: 
[84]: /trailofbits/manticore/wiki/Getting-Started-EVM
[85]: 
[86]: /trailofbits/manticore/wiki/Getting-Started-Native
[87]: 
[88]: /trailofbits/manticore/wiki/Hacking-on-Manticore
[89]: 
[90]: /trailofbits/manticore/wiki/Ideas
[91]: 
[92]: /trailofbits/manticore/wiki/Plugins
[93]: 
[94]: /trailofbits/manticore/wiki/Python2-to-3-Compatibility-Pitfalls
[95]: 
[96]: /trailofbits/manticore/wiki/Redis
[97]: 
[98]: /trailofbits/manticore/wiki/Releasing-Manticore
[99]: 
[100]: /trailofbits/manticore/wiki/Tutorial
[101]: 
[102]: /trailofbits/manticore/wiki/Tutorial:-Adding-Constraints
[103]: 
[104]: /trailofbits/manticore/wiki/Tutorial:-Exercise
[105]: 
[106]: /trailofbits/manticore/wiki/Tutorial:-Getting-Throwing-Path
[107]: 
[108]: /trailofbits/manticore/wiki/Tutorial:-Running-under-Manticore
[109]: 
[110]: /trailofbits/manticore/wiki/What's-in-the-workspace%3F
[111]: 
[112]: /trailofbits/manticore/wiki/Wiki-Page-by-Swapnil_Kothawade-for-POC
[113]: 
[114]: /trailofbits/manticore/wiki/_Sidebar/_edit
[115]: https://github.com/trailofbits/manticore/wiki
[116]: https://github.com/trailofbits/manticore/wiki/Getting-Started-Native
[117]: https://github.com/trailofbits/manticore/wiki/Getting-Started-EVM
[118]: https://github.com/trailofbits/manticore/wiki/Plugins
[119]: https://github.com/trailofbits/manticore/wiki/Tutorial
[120]: https://github.com/trailofbits/manticore/wiki/Tutorial:-Running-under-Manticore
[121]: https://github.com/trailofbits/manticore/wiki/Tutorial:-Getting-Throwing-Path
[122]: https://github.com/trailofbits/manticore/wiki/Tutorial:-Adding-Constraints
[123]: https://github.com/trailofbits/manticore/wiki/Tutorial:-Exercise
[124]: https://github.com/trailofbits/manticore/wiki/What's-in-the-workspace%3F
[125]: https://github.com/trailofbits/manticore/wiki/Analysis-Design-Choices
[126]: https://github.com/trailofbits/manticore/wiki/Analysis-Tips
[127]: https://github.com/trailofbits/manticore/wiki/Docker-notes
[128]: https://github.com/trailofbits/manticore/wiki/Ethereum-Detectors
[129]: https://github.com/trailofbits/manticore/wiki/Hacking-on-Manticore
[130]: https://github.com/trailofbits/manticore/wiki/Ideas
[131]: https://github.com/trailofbits/manticore/wiki/Releasing-Manticore
