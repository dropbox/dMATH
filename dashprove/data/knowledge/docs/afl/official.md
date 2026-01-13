# AFL++ Overview

AFLplusplus is the daughter of the [American Fuzzy Lop][1] fuzzer by Michał “lcamtuf” Zalewski and
was created initially to incorporate all the best features developed in the years for the fuzzers in
the AFL family and not merged in AFL cause it is not updated since November 2017.

[screen1]

The AFL++ fuzzing framework includes the following:

* A fuzzer with many mutators and configurations: afl-fuzz.
* Different source code instrumentation modules: LLVM mode, afl-as, GCC plugin.
* Different binary code instrumentation modules: QEMU mode, Unicorn mode, QBDI mode.
* Utilities for testcase/corpus minimization: afl-tmin, afl-cmin.
* Helper libraries: libtokencap, libdislocator, libcompcov.

It includes a lot of changes, optimizations and new features respect to AFL like the AFLfast power
schedules, QEMU 5.1 upgrade with CompareCoverage, MOpt mutators, InsTrim instrumentation and a lot
more.

See the [Features][2] page.

If you are a student or enthusiast developer and want to contribute, we have an [idea list][3] what
would be cool to have! :-)

If you want to acknoledge our work and the derived works by the academic community in your paper,
see the [Papers][4] page.

It is maintained by Marc “van Hauser” Heuse [mh@mh-sec.de][5], Heiko “hexcoder-” Eißfeldt
[heiko.eissfeldt@hexco.de][6], Andrea Fioraldi [andreafioraldi@gmail.com][7] and Dominik Maier
[mail@dmnk.co][8].

Check out the GitHub repository [here][9].

## Trophies

* VLC
  
  * [CVE-2019-14437][10], [CVE-2019-14438][11], [CVE-2019-14498][12], [CVE-2019-14533][13],
    [CVE-2019-14534][14], [CVE-2019-14535][15], [CVE-2019-14776][16], [CVE-2019-14777][17],
    [CVE-2019-14778][18], [CVE-2019-14779][19], [CVE-2019-14970][20] by Antonio Morales ([GitHub
    Security Lab][21])
* Sqlite
  
  * [CVE-2019-16168][22] by Xingwei Lin (Ant-Financial Light-Year Security Lab)
* Vim
  
  * [CVE-2019-20079][23] by Dhiraj ([blog][24])
* Pure-FTPd
  
  * [CVE-2019-20176][25], [CVE-2020-9274][26], [CVE-2020-9365][27] by Antonio Morales ([GitHub
    Security Lab][28])
* Bftpd
  
  * [CVE-2020-6162][29], [CVE-2020-6835][30] by Antonio Morales ([GitHub Security Lab][31])
* Tcpdump
  
  * [CVE-2020-8036][32] by Reza Mirzazade
* ProFTPd
  
  * [CVE-2020-9272][33], [CVE-2020-9273][34] by Antonio Morales ([GitHub Security Lab][35])
* Gifsicle
  
  * [Issue 130][36] by Ashish Kunwar
* FFmpeg
  
  * [Ticket 8592][37], [Ticket 8593][38], [Ticket 8594][39], [Ticket 8596][40] by Andrea Fioraldi
  * [Ticket 9099][41] by Qiuhao Li
* Glibc
  
  * [Bug 25933][42] by David Mendenhall
* FreeRDP
  
  * [CVE-2020-11095][43], [CVE-2020-11096][44], [CVE-2020-11097][45], [CVE-2020-11098][46],
    [CVE-2020-11099][47], [CVE-2020-13397][48], [CVE-2020-13398][49], [CVE-2020-4030][50],
    [CVE-2020-4031][51], [CVE-2020-4032][52], [CVE-2020-4033][53] by Antonio Morales ([GitHub
    Security Lab][54])
* GNOME
  
  * [Libxps issue 3][55] by Qiuhao Li
* QEMU
  
  * [CVE-2020-29129][56], [CVE-2020-29130][57] by Qiuhao Li
* GNU coreutils
  
  * [Bug 1919775][58] by Qiuhao Li
* PostgreSQL
  
  * [Crash while parsing zero-symbols in jsonb string][59] by [Nikolay Shaplov][60] (Postgres
    Professional)
  * [Bug #18214][61], [Bug #17962][62] `poly_contain` operation works almost forever (using
    [LibBlobStamper][63]) by [Nikolay Shaplov][64] (Postgres Professional)
* Node.js
  
  * [Bug #41949][65], [Bug #46223][66] by [Alexander Shvedov][67]
* libjxl
  
  * [Bug #2100][68] by [Alexander Shvedov][69]
* Perl
  
  * [Bug #20733][70] by [Alexander Shvedov][71]
* zlog
  
  * [CVE-2024-22857][72] ([Details][73]) by [Faran Abdullah][74]

## Sponsoring

We always need servers with many cores for testing various changes for the efficiency. If you want
to sponsor a server with more than 20 cores - contact us! :-)

Current sponsors:

* [Fuzzing IO][75] is sponsoring a 24 core server for one year, thank you! [screen1]

[1]: http://lcamtuf.coredump.cx/afl/
[2]: /features/
[3]: https://github.com/AFLplusplus/AFLplusplus/blob/master/docs/ideas.md
[4]: /papers/
[5]: mailto:mh@mh-sec.de
[6]: mailto:heiko.eissfeldt@hexco.de
[7]: mailto:andreafioraldi@gmail.com
[8]: mailto:mail@dmnk.co
[9]: https://github.com/AFLplusplus/AFLplusplus
[10]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14437
[11]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14438
[12]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14498
[13]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14533
[14]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14534
[15]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14535
[16]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14776
[17]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14777
[18]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14778
[19]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14779
[20]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14970
[21]: https://securitylab.github.com/research/vlc-vulnerability-heap-overflow
[22]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-16168
[23]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-20079
[24]: https://www.inputzero.io/2020/03/fuzzing-vim.html
[25]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-20176
[26]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-14437
[27]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9365
[28]: https://securitylab.github.com/research/fuzzing-sockets-FTP
[29]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-6162
[30]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-6835
[31]: https://securitylab.github.com/research/fuzzing-sockets-FTP
[32]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-8036
[33]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9272
[34]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9273
[35]: https://securitylab.github.com/research/fuzzing-sockets-FTP
[36]: https://github.com/kohler/gifsicle/issues/130
[37]: https://trac.ffmpeg.org/ticket/8592
[38]: https://trac.ffmpeg.org/ticket/8593
[39]: https://trac.ffmpeg.org/ticket/8594
[40]: https://trac.ffmpeg.org/ticket/8596
[41]: https://trac.ffmpeg.org/ticket/9099
[42]: https://sourceware.org/bugzilla/show_bug.cgi?id=25933
[43]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11095
[44]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11096
[45]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11097
[46]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11098
[47]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11099
[48]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13397
[49]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13398
[50]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-4030
[51]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-4031
[52]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-4032
[53]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-4033
[54]: https://securitylab.github.com/research/fuzzing-sockets-FreeRDP
[55]: https://gitlab.gnome.org/GNOME/libgxps/-/issues/3
[56]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-29129
[57]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-29130
[58]: https://bugzilla.redhat.com/show_bug.cgi?id=1919775
[59]: https://www.postgresql.org/message-id/7332649.x5DLKWyVIX%40thinkpad-pgpro
[60]: https://gitlab.com/dhyannataraj
[61]: https://www.postgresql.org/message-id/flat/18214-891f77caa80a35cc%40postgresql.org
[62]: https://www.postgresql.org/message-id/17962-4f00b6f26724858d%40postgresql.org
[63]: https://github.com/postgrespro/libblobstamper
[64]: https://gitlab.com/dhyannataraj
[65]: https://github.com/nodejs/node/issues/41949
[66]: https://github.com/nodejs/node/issues/46223
[67]: https://github.com/a-shvedov
[68]: https://github.com/libjxl/libjxl/issues/2100
[69]: https://github.com/a-shvedov
[70]: https://github.com/Perl/perl5/issues/20733
[71]: https://github.com/a-shvedov
[72]: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-22857
[73]: https://www.ebryx.com/blogs/arbitrary-code-execution-in-zlog-cve-2024-22857
[74]: https://github.com/faran1512
[75]: https://www.fuzzing.io
