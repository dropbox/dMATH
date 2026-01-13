# Welcome to angr’s documentation![¶][1]

Welcome to angr’s documentation! This documentation is intended to be a guide for learning angr, as
well as a reference for the API. If you’re new to angr,

The angr team maintains a number of libraries that are used as part of angr. These libraries are:

* [archinfo][2] - Information about CPU architectures
* [pyvex][3] - Python bindings to the VEX IR
* [pypcode][4] - Python bindings to the Pcode IR
* [cle][5] - Many-platform binary loader
* [claripy][6] - Solver abstraction layer

angr also has a GUI! Check out [angr-management][7].

* [Introduction][8]
  
  * [Getting Support][9]
  * [Citing angr][10]
  * [Going further][11]
* [Getting Started][12]
  
  * [Installing angr][13]
  * [Reporting Bugs][14]
  * [Developing angr][15]
  * [Help Wanted][16]
* [Core Concepts][17]
  
  * [Core Concepts][18]
  * [Loading a Binary][19]
  * [Symbolic Expressions and Constraint Solving][20]
  * [Machine State - memory, registers, and so on][21]
  * [Simulation Managers][22]
  * [Simulation and Instrumentation][23]
  * [Analyses][24]
  * [Symbolic Execution][25]
  * [A final word of advice][26]
* [Build-in Analyses][27]
  
  * [Control-flow Graph Recovery (CFG)][28]
  * [Backward Slicing][29]
  * [Identifier][30]
  * [angr Decompiler][31]
* [Advanced Topics][32]
  
  * [Gotchas when using angr][33]
  * [Understanding the Execution Pipeline][34]
  * [What’s Up With Mixins, Anyway?][35]
  * [Optimization considerations][36]
  * [Working with File System, Sockets, and Pipes][37]
  * [Intermediate Representation][38]
  * [Working with Data and Conventions][39]
  * [Solver Engine][40]
  * [Symbolic memory addressing][41]
  * [Java Support][42]
  * [Debug variable resolution][43]
  * [Variable visibility][44]
* [Extending angr][45]
  
  * [Hooks and SimProcedures][46]
  * [State Plugins][47]
  * [Extending the Environment Model][48]
  * [Writing Analyses][49]
* [angr examples][50]
  
  * [Introduction][51]
  * [Reversing][52]
  * [Vulnerability Discovery][53]
  * [Exploitation][54]
* [Frequently Asked Questions][55]
  
  * [Why is it named angr?][56]
  * [How should “angr” be stylized?][57]
  * [Why isn’t symbolic execution doing the thing I want?][58]
  * [How can I get diagnostic information about what angr is doing?][59]
  * [Why is angr so slow?][60]
  * [How do I find bugs using angr?][61]
  * [Why did you choose VEX instead of another IR (such as LLVM, REIL, BAP, etc)?][62]
  * [Why are some ARM addresses off-by-one?][63]
  * [How do I serialize angr objects?][64]
  * [What does `UnsupportedIROpError("floating point support disabled")` mean?][65]
  * [Why is angr’s CFG different from IDA’s?][66]
  * [Why do I get incorrect register values when reading from a state during a SimInspect
    breakpoint?][67]
* [Appendix][68]
  
  * [Cheatsheet][69]
  * [List of Claripy Operations][70]
  * [List of State Options][71]
  * [CTF Challenge Examples][72]
  * [Changelog][73]
  * [Migrating to angr 9.1][74]
  * [Migrating to angr 8][75]
  * [Migrating to angr 7][76]
* [API Reference][77]
  
  * [`BP`][78]
  * [`Analysis`][79]
  * [`AngrAnalysisError`][80]
  * [`AngrAnnotatedCFGError`][81]
  * [`AngrAssemblyError`][82]
  * [`AngrBackwardSlicingError`][83]
  * [`AngrBladeError`][84]
  * [`AngrBladeSimProcError`][85]
  * [`AngrCFGError`][86]
  * [`AngrCallableError`][87]
  * [`AngrCallableMultistateError`][88]
  * [`AngrCorruptDBError`][89]
  * [`AngrDBError`][90]
  * [`AngrDDGError`][91]
  * [`AngrDataGraphError`][92]
  * [`AngrDecompilationError`][93]
  * [`AngrDelayJobNotice`][94]
  * [`AngrDirectorError`][95]
  * [`AngrError`][96]
  * [`AngrExitError`][97]
  * [`AngrExplorationTechniqueError`][98]
  * [`AngrExplorerError`][99]
  * [`AngrForwardAnalysisError`][100]
  * [`AngrIncompatibleDBError`][101]
  * [`AngrIncongruencyError`][102]
  * [`AngrInvalidArgumentError`][103]
  * [`AngrJobMergingFailureNotice`][104]
  * [`AngrJobWideningFailureNotice`][105]
  * [`AngrLifterError`][106]
  * [`AngrLoopAnalysisError`][107]
  * [`AngrMissingTypeError`][108]
  * [`AngrNoPluginError`][109]
  * [`AngrPathError`][110]
  * [`AngrRuntimeError`][111]
  * [`AngrSimOSError`][112]
  * [`AngrSkipJobNotice`][113]
  * [`AngrSurveyorError`][114]
  * [`AngrSyscallError`][115]
  * [`AngrTracerError`][116]
  * [`AngrTypeError`][117]
  * [`AngrUnsupportedSyscallError`][118]
  * [`AngrVFGError`][119]
  * [`AngrVFGRestartAnalysisNotice`][120]
  * [`AngrValueError`][121]
  * [`AngrVaultError`][122]
  * [`Blade`][123]
  * [`Block`][124]
  * [`Emulator`][125]
  * [`EmulatorStopReason`][126]
  * [`ExplorationTechnique`][127]
  * [`KnowledgeBase`][128]
  * [`PTChunk`][129]
  * [`PathUnreachableError`][130]
  * [`PointerWrapper`][131]
  * [`Project`][132]
  * [`Server`][133]
  * [`SimAbstractMemoryError`][134]
  * [`SimActionError`][135]
  * [`SimCC`][136]
  * [`SimCCError`][137]
  * [`SimCCallError`][138]
  * [`SimConcreteBreakpointError`][139]
  * [`SimConcreteMemoryError`][140]
  * [`SimConcreteRegisterError`][141]
  * [`SimEmptyCallStackError`][142]
  * [`SimEngineError`][143]
  * [`SimError`][144]
  * [`SimEventError`][145]
  * [`SimException`][146]
  * [`SimExpressionError`][147]
  * [`SimFastMemoryError`][148]
  * [`SimFastPathError`][149]
  * [`SimFile`][150]
  * [`SimFileBase`][151]
  * [`SimFileDescriptor`][152]
  * [`SimFileDescriptorDuplex`][153]
  * [`SimFileError`][154]
  * [`SimFileStream`][155]
  * [`SimFilesystemError`][156]
  * [`SimHeapBrk`][157]
  * [`SimHeapError`][158]
  * [`SimHeapPTMalloc`][159]
  * [`SimHostFilesystem`][160]
  * [`SimIRSBError`][161]
  * [`SimIRSBNoDecodeError`][162]
  * [`SimMemoryAddressError`][163]
  * [`SimMemoryError`][164]
  * [`SimMemoryLimitError`][165]
  * [`SimMemoryMissingError`][166]
  * [`SimMergeError`][167]
  * [`SimMissingTempError`][168]
  * [`SimMount`][169]
  * [`SimOS`][170]
  * [`SimOperationError`][171]
  * [`SimPackets`][172]
  * [`SimPacketsStream`][173]
  * [`SimPosixError`][174]
  * [`SimProcedure`][175]
  * [`SimProcedureArgumentError`][176]
  * [`SimProcedureError`][177]
  * [`SimRegionMapError`][178]
  * [`SimReliftException`][179]
  * [`SimSegfaultError`][180]
  * [`SimSegfaultException`][181]
  * [`SimShadowStackError`][182]
  * [`SimSlicerError`][183]
  * [`SimSolverError`][184]
  * [`SimSolverModeError`][185]
  * [`SimSolverOptionError`][186]
  * [`SimState`][187]
  * [`SimStateError`][188]
  * [`SimStateOptionsError`][189]
  * [`SimStatePlugin`][190]
  * [`SimStatementError`][191]
  * [`SimSymbolicFilesystemError`][192]
  * [`SimTranslationError`][193]
  * [`SimUCManagerAllocationError`][194]
  * [`SimUCManagerError`][195]
  * [`SimUnicornError`][196]
  * [`SimUnicornSymbolic`][197]
  * [`SimUnicornUnsupport`][198]
  * [`SimUninitializedAccessError`][199]
  * [`SimUnsatError`][200]
  * [`SimUnsupportedError`][201]
  * [`SimValueError`][202]
  * [`SimZeroDivisionException`][203]
  * [`SimulationManager`][204]
  * [`SimulationManagerError`][205]
  * [`StateHierarchy`][206]
  * [`TracerEnvironmentError`][207]
  * [`UnsupportedCCallError`][208]
  * [`UnsupportedDirtyError`][209]
  * [`UnsupportedIRExprError`][210]
  * [`UnsupportedIROpError`][211]
  * [`UnsupportedIRStmtError`][212]
  * [`UnsupportedNodeTypeError`][213]
  * [`UnsupportedSyscallError`][214]
  * [`default_cc()`][215]
  * [`load_shellcode()`][216]
  * [`register_analysis()`][217]
  * [Project][218]
  * [Plugin Ecosystem][219]
  * [Program State][220]
  * [Storage][221]
  * [Memory Mixins][222]
  * [Concretization Strategies][223]
  * [Simulation Manager][224]
  * [Exploration Techniques][225]
  * [Simulation Engines][226]
  * [Simulation Logging][227]
  * [Procedures][228]
  * [Calling Conventions and Types][229]
  * [Knowledge Base][230]
  * [Serialization][231]
  * [Analysis][232]
  * [SimOS][233]
  * [Function Signature Matching][234]
  * [Utils][235]
  * [Errors][236]
  * [Distributed analysis][237]
  * [angr Intermediate Language][238]

# Indices and tables[¶][239]

* [Index][240]
* [Module Index][241]
* [Search Page][242]

[1]: #welcome-to-angr-s-documentation
[2]: https://api.angr.io/projects/archinfo/en/latest/
[3]: https://api.angr.io/projects/pyvex/en/latest/
[4]: https://api.angr.io/projects/pypcode/en/latest/
[5]: https://api.angr.io/projects/cle/en/latest/
[6]: https://api.angr.io/projects/claripy/en/latest/
[7]: https://github.com/angr/angr-management/
[8]: quickstart.html
[9]: quickstart.html#getting-support
[10]: quickstart.html#citing-angr
[11]: quickstart.html#going-further
[12]: getting-started/index.html
[13]: getting-started/installing.html
[14]: getting-started/developing.html
[15]: getting-started/developing.html#developing-angr
[16]: getting-started/helpwanted.html
[17]: core-concepts/index.html
[18]: core-concepts/toplevel.html
[19]: core-concepts/loading.html
[20]: core-concepts/solver.html
[21]: core-concepts/states.html
[22]: core-concepts/pathgroups.html
[23]: core-concepts/simulation.html
[24]: core-concepts/analyses.html
[25]: core-concepts/symbolic.html
[26]: core-concepts/be_creative.html
[27]: analyses/index.html
[28]: analyses/cfg.html
[29]: analyses/backward_slice.html
[30]: analyses/identifier.html
[31]: analyses/decompiler.html
[32]: advanced-topics/index.html
[33]: advanced-topics/gotchas.html
[34]: advanced-topics/pipeline.html
[35]: advanced-topics/mixins.html
[36]: advanced-topics/speed.html
[37]: advanced-topics/file_system.html
[38]: advanced-topics/ir.html
[39]: advanced-topics/structured_data.html
[40]: advanced-topics/claripy.html
[41]: advanced-topics/concretization_strategies.html
[42]: advanced-topics/java_support.html
[43]: advanced-topics/debug_var.html
[44]: advanced-topics/debug_var.html#variable-visibility
[45]: extending-angr/index.html
[46]: extending-angr/simprocedures.html
[47]: extending-angr/state_plugins.html
[48]: extending-angr/environment.html
[49]: extending-angr/analysis_writing.html
[50]: examples.html
[51]: examples.html#introduction
[52]: examples.html#reversing
[53]: examples.html#vulnerability-discovery
[54]: examples.html#exploitation
[55]: faq.html
[56]: faq.html#why-is-it-named-angr
[57]: faq.html#how-should-angr-be-stylized
[58]: faq.html#why-isn-t-symbolic-execution-doing-the-thing-i-want
[59]: faq.html#how-can-i-get-diagnostic-information-about-what-angr-is-doing
[60]: faq.html#why-is-angr-so-slow
[61]: faq.html#how-do-i-find-bugs-using-angr
[62]: faq.html#why-did-you-choose-vex-instead-of-another-ir-such-as-llvm-reil-bap-etc
[63]: faq.html#why-are-some-arm-addresses-off-by-one
[64]: faq.html#how-do-i-serialize-angr-objects
[65]: faq.html#what-does-unsupportediroperror-floating-point-support-disabled-mean
[66]: faq.html#why-is-angr-s-cfg-different-from-ida-s
[67]: faq.html#why-do-i-get-incorrect-register-values-when-reading-from-a-state-during-a-siminspect-
breakpoint
[68]: appendix/index.html
[69]: appendix/cheatsheet.html
[70]: appendix/ops.html
[71]: appendix/options.html
[72]: appendix/more-examples.html
[73]: appendix/changelog.html
[74]: appendix/migration-9.1.html
[75]: appendix/migration-8.html
[76]: appendix/migration-7.html
[77]: api.html
[78]: api.html#angr.BP
[79]: api.html#angr.Analysis
[80]: api.html#angr.AngrAnalysisError
[81]: api.html#angr.AngrAnnotatedCFGError
[82]: api.html#angr.AngrAssemblyError
[83]: api.html#angr.AngrBackwardSlicingError
[84]: api.html#angr.AngrBladeError
[85]: api.html#angr.AngrBladeSimProcError
[86]: api.html#angr.AngrCFGError
[87]: api.html#angr.AngrCallableError
[88]: api.html#angr.AngrCallableMultistateError
[89]: api.html#angr.AngrCorruptDBError
[90]: api.html#angr.AngrDBError
[91]: api.html#angr.AngrDDGError
[92]: api.html#angr.AngrDataGraphError
[93]: api.html#angr.AngrDecompilationError
[94]: api.html#angr.AngrDelayJobNotice
[95]: api.html#angr.AngrDirectorError
[96]: api.html#angr.AngrError
[97]: api.html#angr.AngrExitError
[98]: api.html#angr.AngrExplorationTechniqueError
[99]: api.html#angr.AngrExplorerError
[100]: api.html#angr.AngrForwardAnalysisError
[101]: api.html#angr.AngrIncompatibleDBError
[102]: api.html#angr.AngrIncongruencyError
[103]: api.html#angr.AngrInvalidArgumentError
[104]: api.html#angr.AngrJobMergingFailureNotice
[105]: api.html#angr.AngrJobWideningFailureNotice
[106]: api.html#angr.AngrLifterError
[107]: api.html#angr.AngrLoopAnalysisError
[108]: api.html#angr.AngrMissingTypeError
[109]: api.html#angr.AngrNoPluginError
[110]: api.html#angr.AngrPathError
[111]: api.html#angr.AngrRuntimeError
[112]: api.html#angr.AngrSimOSError
[113]: api.html#angr.AngrSkipJobNotice
[114]: api.html#angr.AngrSurveyorError
[115]: api.html#angr.AngrSyscallError
[116]: api.html#angr.AngrTracerError
[117]: api.html#angr.AngrTypeError
[118]: api.html#angr.AngrUnsupportedSyscallError
[119]: api.html#angr.AngrVFGError
[120]: api.html#angr.AngrVFGRestartAnalysisNotice
[121]: api.html#angr.AngrValueError
[122]: api.html#angr.AngrVaultError
[123]: api.html#angr.Blade
[124]: api.html#angr.Block
[125]: api.html#angr.Emulator
[126]: api.html#angr.EmulatorStopReason
[127]: api.html#angr.ExplorationTechnique
[128]: api.html#angr.KnowledgeBase
[129]: api.html#angr.PTChunk
[130]: api.html#angr.PathUnreachableError
[131]: api.html#angr.PointerWrapper
[132]: api.html#angr.Project
[133]: api.html#angr.Server
[134]: api.html#angr.SimAbstractMemoryError
[135]: api.html#angr.SimActionError
[136]: api.html#angr.SimCC
[137]: api.html#angr.SimCCError
[138]: api.html#angr.SimCCallError
[139]: api.html#angr.SimConcreteBreakpointError
[140]: api.html#angr.SimConcreteMemoryError
[141]: api.html#angr.SimConcreteRegisterError
[142]: api.html#angr.SimEmptyCallStackError
[143]: api.html#angr.SimEngineError
[144]: api.html#angr.SimError
[145]: api.html#angr.SimEventError
[146]: api.html#angr.SimException
[147]: api.html#angr.SimExpressionError
[148]: api.html#angr.SimFastMemoryError
[149]: api.html#angr.SimFastPathError
[150]: api.html#angr.SimFile
[151]: api.html#angr.SimFileBase
[152]: api.html#angr.SimFileDescriptor
[153]: api.html#angr.SimFileDescriptorDuplex
[154]: api.html#angr.SimFileError
[155]: api.html#angr.SimFileStream
[156]: api.html#angr.SimFilesystemError
[157]: api.html#angr.SimHeapBrk
[158]: api.html#angr.SimHeapError
[159]: api.html#angr.SimHeapPTMalloc
[160]: api.html#angr.SimHostFilesystem
[161]: api.html#angr.SimIRSBError
[162]: api.html#angr.SimIRSBNoDecodeError
[163]: api.html#angr.SimMemoryAddressError
[164]: api.html#angr.SimMemoryError
[165]: api.html#angr.SimMemoryLimitError
[166]: api.html#angr.SimMemoryMissingError
[167]: api.html#angr.SimMergeError
[168]: api.html#angr.SimMissingTempError
[169]: api.html#angr.SimMount
[170]: api.html#angr.SimOS
[171]: api.html#angr.SimOperationError
[172]: api.html#angr.SimPackets
[173]: api.html#angr.SimPacketsStream
[174]: api.html#angr.SimPosixError
[175]: api.html#angr.SimProcedure
[176]: api.html#angr.SimProcedureArgumentError
[177]: api.html#angr.SimProcedureError
[178]: api.html#angr.SimRegionMapError
[179]: api.html#angr.SimReliftException
[180]: api.html#angr.SimSegfaultError
[181]: api.html#angr.SimSegfaultException
[182]: api.html#angr.SimShadowStackError
[183]: api.html#angr.SimSlicerError
[184]: api.html#angr.SimSolverError
[185]: api.html#angr.SimSolverModeError
[186]: api.html#angr.SimSolverOptionError
[187]: api.html#angr.SimState
[188]: api.html#angr.SimStateError
[189]: api.html#angr.SimStateOptionsError
[190]: api.html#angr.SimStatePlugin
[191]: api.html#angr.SimStatementError
[192]: api.html#angr.SimSymbolicFilesystemError
[193]: api.html#angr.SimTranslationError
[194]: api.html#angr.SimUCManagerAllocationError
[195]: api.html#angr.SimUCManagerError
[196]: api.html#angr.SimUnicornError
[197]: api.html#angr.SimUnicornSymbolic
[198]: api.html#angr.SimUnicornUnsupport
[199]: api.html#angr.SimUninitializedAccessError
[200]: api.html#angr.SimUnsatError
[201]: api.html#angr.SimUnsupportedError
[202]: api.html#angr.SimValueError
[203]: api.html#angr.SimZeroDivisionException
[204]: api.html#angr.SimulationManager
[205]: api.html#angr.SimulationManagerError
[206]: api.html#angr.StateHierarchy
[207]: api.html#angr.TracerEnvironmentError
[208]: api.html#angr.UnsupportedCCallError
[209]: api.html#angr.UnsupportedDirtyError
[210]: api.html#angr.UnsupportedIRExprError
[211]: api.html#angr.UnsupportedIROpError
[212]: api.html#angr.UnsupportedIRStmtError
[213]: api.html#angr.UnsupportedNodeTypeError
[214]: api.html#angr.UnsupportedSyscallError
[215]: api.html#angr.default_cc
[216]: api.html#angr.load_shellcode
[217]: api.html#angr.register_analysis
[218]: api.html#module-angr.project
[219]: api.html#module-angr.misc.plugins
[220]: api.html#module-angr.sim_state
[221]: api.html#module-angr.storage
[222]: api.html#module-angr.storage.memory_mixins
[223]: api.html#module-angr.concretization_strategies.single
[224]: api.html#module-angr.sim_manager
[225]: api.html#module-angr.exploration_techniques
[226]: api.html#module-angr.engines
[227]: api.html#module-angr.state_plugins.sim_action
[228]: api.html#module-angr.sim_procedure
[229]: api.html#module-angr.calling_conventions
[230]: api.html#module-angr.knowledge_base
[231]: api.html#module-angr.serializable
[232]: api.html#module-angr.analyses
[233]: api.html#module-angr.simos
[234]: api.html#module-angr.flirt
[235]: api.html#module-angr.utils
[236]: api.html#module-angr.errors
[237]: api.html#module-angr.distributed
[238]: api.html#module-angr.ailment
[239]: #indices-and-tables
[240]: genindex.html
[241]: py-modindex.html
[242]: search.html
