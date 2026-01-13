[Yosys -- Yosys Open SYnthesis Suite]
[About][1] [Documentation][2] [F.A.Q.][3] [Screenshots][4] [Download][5] [Links][6] [Commercial
Support and Development][7] [YosysHQ Community Slack][8] [IRC (Libera Chat)][9] [GitHub][10]
Related Projects:
[VlogHammer][11] [YosysJS][12]

# About

Yosys is a framework for Verilog RTL synthesis. It currently has extensive Verilog-2005 support and
provides a basic set of synthesis algorithms for various application domains. Selected features and
typical applications:

* Process almost any synthesizable Verilog-2005 design
* Converting Verilog to BLIF / EDIF/ BTOR / SMT-LIB / simple RTL Verilog / etc.
* Built-in formal methods for checking properties and equivalence
* Mapping to ASIC standard cell libraries (in Liberty File Format)
* Mapping to Xilinx 7-Series and Lattice iCE40 and ECP5 FPGAs
* Foundation and/or front-end for custom flows

Yosys can be adapted to perform any synthesis job by combining the existing passes (algorithms)
using synthesis scripts and adding additional passes as needed by extending the Yosys C++ code base.

Yosys also serves as backend for several tools that use formal methods to reason about designs, such
as [sby][13] for SMT-solver-based formal property checking or [mcy][14] for evaluating the quality
of testbenches with mutation coverage metrics.


Yosys is free software licensed under the [ISC license][15] (a GPL compatible license that is
similar in terms to the MIT license or the 2-clause BSD license).

## Example Usage

Yosys is controlled using synthesis scripts. For example, the following Yosys synthesis script reads
a design (with the top module mytop) from the verilog file mydesign.v, synthesizes it to a
gate-level netlist using the cell library in the Liberty file mycells.lib and writes the synthesized
results as Verilog netlist to synth.v:

# read design
read_verilog mydesign.v

# elaborate design hierarchy
hierarchy -check -top mytop

# the high-level stuff
proc; opt; fsm; opt; memory; opt

# mapping to internal cell library
techmap; opt

# mapping flip-flops to mycells.lib
dfflibmap -liberty mycells.lib

# mapping logic to mycells.lib
abc -liberty mycells.lib

# cleanup
clean

# write synthesized design
write_verilog synth.v

The synth command provides a good default script that can be used as basis for simple synthesis
scripts:

# read design
read_verilog mydesign.v

# generic synthesis
synth -top mytop

# mapping to mycells.lib
dfflibmap -liberty mycells.lib
abc -liberty mycells.lib
clean

# write synthesized design
write_verilog synth.v

See [help synth][16] for details on the synth command.

[1]: about.html
[2]: documentation.html
[3]: faq.html
[4]: screenshots.html
[5]: download.html
[6]: links.html
[7]: commercial.html
[8]: https://join.slack.com/t/yosyshq/shared_invite/zt-oe2nxfpv-BJd_9CZpkk_MoTT0s88GcA
[9]: https://web.libera.chat/#yosys
[10]: https://github.com/YosysHQ/yosys
[11]: vloghammer.html
[12]: yosysjs.html
[13]: https://yosyshq.readthedocs.io/en/latest/tools.html#formal-assertions-based-verification-abv-w
ith-symbiyosys-sby
[14]: https://yosyshq.readthedocs.io/en/latest/tools.html#mutation-coverage-with-yosys-mcy
[15]: http://en.wikipedia.org/wiki/ISC_license
[16]: https://yosys.readthedocs.io/en/latest/cmd/synth.html
