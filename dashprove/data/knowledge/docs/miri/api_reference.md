# Crate miri Copy item path

[Source][1]

## Re-exports[§][2]

*`pub use rustc_const_eval::[interpret][3];`*
*`pub use rustc_const_eval::interpret::[AllocMap][4];`*
*`pub use rustc_const_eval::interpret::[Provenance][5] as _;`*
*`pub use crate::alloc_addresses::[EvalContextExt][6] as _;`*
*`pub use crate::borrow_tracker::stacked_borrows::[EvalContextExt][7] as _;`*
*`pub use crate::borrow_tracker::tree_borrows::[EvalContextExt][8] as _;`*
*`pub use crate::borrow_tracker::[EvalContextExt][9] as _;`*
*`pub use crate::concurrency::data_race::[EvalContextExt][10] as _;`*
*`pub use crate::concurrency::init_once::[EvalContextExt][11] as _;`*
*`pub use crate::concurrency::sync::[EvalContextExt][12] as _;`*
*`pub use crate::concurrency::thread::[EvalContextExt][13] as _;`*
*`pub use crate::diagnostics::[EvalContextExt][14] as _;`*
*`pub use crate::helpers::[EvalContextExt][15] as _;`*
*`pub use crate::helpers::[ToU64][16] as _;`*
*`pub use crate::helpers::[ToUsize][17] as _;`*
*`pub use crate::intrinsics::[EvalContextExt][18] as _;`*
*`pub use crate::operator::[EvalContextExt][19] as _;`*
*`pub use crate::provenance_gc::[EvalContextExt][20] as _;`*
*`pub use crate::shims::env::[EvalContextExt][21] as _;`*
*`pub use crate::shims::foreign_items::[EvalContextExt][22] as _;`*
*`pub use crate::shims::io_error::[EvalContextExt][23] as _;`*
*`pub use crate::shims::io_error::[LibcError][24];`*
*`pub use crate::shims::os_str::[EvalContextExt][25] as _;`*
*`pub use crate::shims::panic::[EvalContextExt][26] as _;`*
*`pub use crate::shims::sig::[EvalContextExt][27] as _;`*
*`pub use crate::shims::time::[EvalContextExt][28] as _;`*
*`pub use crate::shims::unwind::[EvalContextExt][29] as _;`*
*`pub use rustc_const_eval::[interpret][30]::*;`*

## Modules[§][31]

*[alloc][32] 🔒 *
*[alloc_addresses][33] 🔒 *
  This module is responsible for managing the absolute addresses that allocations are located at,
  and for casting between pointers and integers based on those addresses.
*[borrow_tracker][34] 🔒 *
*[clock][35] 🔒 *
*[concurrency][36] 🔒 *
*[data_structures][37] 🔒 *
*[diagnostics][38] 🔒 *
*[eval][39] 🔒 *
  Main evaluator loop and setting up the initial stack frame.
*[helpers][40] 🔒 *
*[intrinsics][41] 🔒 *
*[machine][42] 🔒 *
  Global machine state as well as implementation of the interpreter engine `Machine` trait.
*[math][43] 🔒 *
*[native_lib][44]*
*[operator][45] 🔒 *
*[provenance_gc][46] 🔒 *
*[shims][47] 🔒 *

## Macros[§][48]

*[callback][49]*
  Creates a `DynMachineCallback`:
*[enter_trace_span][50]*
  Enters a [tracing::info_span] only if the “tracing” feature is enabled, otherwise does nothing.
  This calls [rustc_const_eval::enter_trace_span][51] with [MiriMachine][52] as the first argument,
  which will in turn call [MiriMachine::enter_trace_span][53], which takes care of determining at
  compile time whether to trace or not (and supposedly the call is compiled out if tracing is
  disabled). Look at [rustc_const_eval::enter_trace_span][54] for complete documentation, examples
  and tips.
*[shim_sig][55]*
  Construct a `ShimSig` with convenient syntax:
*[shim_sig_arg][56]*
  Helper for `shim_sig!`.

## Structs[§][57]

*[AllocExtra][58]*
  Extra per-allocation data
*[BorTag][59]*
  Tracking pointer provenance
*[CatchUnwindData][60]*
  Holds all of the relevant data for when unwinding hits a `try` frame.
*[CondvarRef][61]*
*[DedupRangeMap][62]*
*[DynSym][63]*
  Type of dynamic symbols (for `dlsym` et al)
*[FrameExtra][64]*
  Extra data stored with each stack frame
*[GenmcConfig][65]*
*[GenmcCtx][66]*
*[InitOnceRef][67]*
*[Instant][68]*
*[Item][69]*
  An item in the per-location borrow stack.
*[LiveAllocs][70]*
*[MiriAllocBytes][71]*
  Allocation bytes that explicitly handle the layout of the data they’re storing. This is necessary
  to interface with native code that accesses the program store in Miri.
*[MiriConfig][72]*
  Configuration needed to spawn a Miri instance.
*[MiriMachine][73]*
  The machine itself.
*[MonoHashMap][74]*
*[MonotonicClock][75]*
  A monotone clock used for `Instant` simulation.
*[MutexRef][76]*
*[PrimitiveLayouts][77]*
  Precomputed layouts of primitive types
*[RwLockRef][78]*
*[Stack][79]*
  Extra per-location state.
*[Stacks][80]*
  Extra per-allocation state.
*[ThreadId][81]*
  A thread identifier.
*[ThreadManager][82]*
  A set of threads.
*[TlsData][83]*
*[Tree][84]*
  Tree structure with both parents and children since we want to be able to traverse the tree
  efficiently in both directions.
*[TreeBorrowsParams][85]*
  Parameters that Tree Borrows can take.

## Enums[§][86]

*[AlignmentCheck][87]*
*[AtomicFenceOrd][88]*
  Valid atomic fence orderings, subset of atomic::Ordering.
*[AtomicReadOrd][89]*
  Valid atomic read orderings, subset of atomic::Ordering.
*[AtomicRwOrd][90]*
  Valid atomic read-write orderings, alias of atomic::Ordering (not non-exhaustive).
*[AtomicWriteOrd][91]*
  Valid atomic write orderings, subset of atomic::Ordering.
*[BacktraceStyle][92]*
*[BlockReason][93]*
  Keeps track of what the thread is blocked on.
*[BorrowTrackerMethod][94]*
  Which borrow tracking method to use
*[EmulateItemResult][95]*
  What needs to be done after emulating an item (a shim or an intrinsic) is done.
*[EnvVars][96]*
*[FloatRoundingErrorMode][97]*
*[IoError][98]*
  A representation of an IO error: either a libc error name, or a host error.
*[IsolatedOp][99]*
*[MiriEntryFnType][100]*
*[MiriMemoryKind][101]*
  Extra memory kinds
*[NonHaltingDiagnostic][102]*
  Miri specific diagnostics
*[Permission][103]*
  Indicates which permission is granted (by this item to some pointers)
*[Provenance][104]*
  Pointer provenance.
*[ProvenanceExtra][105]*
  The “extra” information a pointer has over a regular AllocId.
*[ProvenanceMode][106]*
*[RejectOpWith][107]*
*[TerminationInfo][108]*
  Details of premature program termination.
*[TimeoutAnchor][109]*
  Whether the timeout is relative or absolute.
*[TimeoutClock][110]*
  The clock to use for the timeout you are asking for.
*[UnblockKind][111]*
  The argument type for the “unblock” callback, indicating why the thread got unblocked.
*[ValidationMode][112]*

## Constants[§][113]

*[MAX_CPUS][114]*
  The maximum number of CPUs supported by miri.
*[MIRI_DEFAULT_ARGS][115]*
  Insert rustc arguments at the beginning of the argument list that Miri wants to be set per
  default, for maximal validation power. Also disable the MIR pass that inserts an alignment check
  on every pointer dereference. Miri does that too, and with a better error message.

## Traits[§][116]

*[MachineCallback][117]*
  Trait for callbacks handling asynchronous machine operations.
*[MiriInterpCxExt][118]*
  A little trait that’s useful to be inherited by extension traits.
*[VisitProvenance][119]*

## Functions[§][120]

*[create_ecx][121]*
  Returns a freshly created `InterpCx`. Public because this is also used by `priroda`.
*[eval_entry][122]*
  Evaluates the entry function specified by `entry_id`. Returns `Some(return_code)` if program
  execution completed. Returns `None` if an evaluation error occurred.
*[report_result][123]*
  Report the result of a Miri execution.
*[run_genmc_mode][124]*

## Type Aliases[§][125]

*[DynMachineCallback][126]*
  Type alias for boxed machine callbacks with generic argument type.
*[DynUnblockCallback][127]*
  Type alias for unblock callbacks, i.e. machine callbacks invoked when a thread gets unblocked.
*[FnArg][128]*
*[ImmTy][129]*
*[MPlaceTy][130]*
*[MemoryKind][131]*
*[MiriInterpCx][132]*
  A rustc InterpCx for Miri.
*[OpTy][133]*
*[PlaceTy][134]*
*[Pointer][135]*
*[Scalar][136]*
*[StackEmptyCallback][137]*
*[StrictPointer][138]*
*[VisitWith][139]*

[1]: ../src/miri/lib.rs.html#1-182
[2]: #reexports
[3]: ../rustc_const_eval/interpret/index.html
[4]: ../rustc_const_eval/interpret/machine/trait.AllocMap.html
[5]: ../rustc_middle/mir/interpret/pointer/trait.Provenance.html
[6]: alloc_addresses/trait.EvalContextExt.html
[7]: borrow_tracker/stacked_borrows/trait.EvalContextExt.html
[8]: borrow_tracker/tree_borrows/trait.EvalContextExt.html
[9]: borrow_tracker/trait.EvalContextExt.html
[10]: concurrency/data_race/trait.EvalContextExt.html
[11]: concurrency/init_once/trait.EvalContextExt.html
[12]: concurrency/sync/trait.EvalContextExt.html
[13]: concurrency/thread/trait.EvalContextExt.html
[14]: diagnostics/trait.EvalContextExt.html
[15]: helpers/trait.EvalContextExt.html
[16]: helpers/trait.ToU64.html
[17]: helpers/trait.ToUsize.html
[18]: intrinsics/trait.EvalContextExt.html
[19]: operator/trait.EvalContextExt.html
[20]: provenance_gc/trait.EvalContextExt.html
[21]: shims/env/trait.EvalContextExt.html
[22]: shims/foreign_items/trait.EvalContextExt.html
[23]: shims/io_error/trait.EvalContextExt.html
[24]: enum.IoError.html#variant.LibcError
[25]: shims/os_str/trait.EvalContextExt.html
[26]: shims/panic/trait.EvalContextExt.html
[27]: shims/sig/trait.EvalContextExt.html
[28]: shims/time/trait.EvalContextExt.html
[29]: shims/unwind/trait.EvalContextExt.html
[30]: ../rustc_const_eval/interpret/index.html
[31]: #modules
[32]: alloc/index.html
[33]: alloc_addresses/index.html
[34]: borrow_tracker/index.html
[35]: clock/index.html
[36]: concurrency/index.html
[37]: data_structures/index.html
[38]: diagnostics/index.html
[39]: eval/index.html
[40]: helpers/index.html
[41]: intrinsics/index.html
[42]: machine/index.html
[43]: math/index.html
[44]: native_lib/index.html
[45]: operator/index.html
[46]: provenance_gc/index.html
[47]: shims/index.html
[48]: #macros
[49]: macro.callback.html
[50]: macro.enter_trace_span.html
[51]: ../rustc_const_eval/macro.enter_trace_span.html
[52]: struct.MiriMachine.html
[53]: struct.MiriMachine.html#method.enter_trace_span
[54]: ../rustc_const_eval/macro.enter_trace_span.html
[55]: macro.shim_sig.html
[56]: macro.shim_sig_arg.html
[57]: #structs
[58]: struct.AllocExtra.html
[59]: struct.BorTag.html
[60]: struct.CatchUnwindData.html
[61]: struct.CondvarRef.html
[62]: struct.DedupRangeMap.html
[63]: struct.DynSym.html
[64]: struct.FrameExtra.html
[65]: struct.GenmcConfig.html
[66]: struct.GenmcCtx.html
[67]: struct.InitOnceRef.html
[68]: struct.Instant.html
[69]: struct.Item.html
[70]: struct.LiveAllocs.html
[71]: struct.MiriAllocBytes.html
[72]: struct.MiriConfig.html
[73]: struct.MiriMachine.html
[74]: struct.MonoHashMap.html
[75]: struct.MonotonicClock.html
[76]: struct.MutexRef.html
[77]: struct.PrimitiveLayouts.html
[78]: struct.RwLockRef.html
[79]: struct.Stack.html
[80]: struct.Stacks.html
[81]: struct.ThreadId.html
[82]: struct.ThreadManager.html
[83]: struct.TlsData.html
[84]: struct.Tree.html
[85]: struct.TreeBorrowsParams.html
[86]: #enums
[87]: enum.AlignmentCheck.html
[88]: enum.AtomicFenceOrd.html
[89]: enum.AtomicReadOrd.html
[90]: enum.AtomicRwOrd.html
[91]: enum.AtomicWriteOrd.html
[92]: enum.BacktraceStyle.html
[93]: enum.BlockReason.html
[94]: enum.BorrowTrackerMethod.html
[95]: enum.EmulateItemResult.html
[96]: enum.EnvVars.html
[97]: enum.FloatRoundingErrorMode.html
[98]: enum.IoError.html
[99]: enum.IsolatedOp.html
[100]: enum.MiriEntryFnType.html
[101]: enum.MiriMemoryKind.html
[102]: enum.NonHaltingDiagnostic.html
[103]: enum.Permission.html
[104]: enum.Provenance.html
[105]: enum.ProvenanceExtra.html
[106]: enum.ProvenanceMode.html
[107]: enum.RejectOpWith.html
[108]: enum.TerminationInfo.html
[109]: enum.TimeoutAnchor.html
[110]: enum.TimeoutClock.html
[111]: enum.UnblockKind.html
[112]: enum.ValidationMode.html
[113]: #constants
[114]: constant.MAX_CPUS.html
[115]: constant.MIRI_DEFAULT_ARGS.html
[116]: #traits
[117]: trait.MachineCallback.html
[118]: trait.MiriInterpCxExt.html
[119]: trait.VisitProvenance.html
[120]: #functions
[121]: fn.create_ecx.html
[122]: fn.eval_entry.html
[123]: fn.report_result.html
[124]: fn.run_genmc_mode.html
[125]: #types
[126]: type.DynMachineCallback.html
[127]: type.DynUnblockCallback.html
[128]: type.FnArg.html
[129]: type.ImmTy.html
[130]: type.MPlaceTy.html
[131]: type.MemoryKind.html
[132]: type.MiriInterpCx.html
[133]: type.OpTy.html
[134]: type.PlaceTy.html
[135]: type.Pointer.html
[136]: type.Scalar.html
[137]: type.StackEmptyCallback.html
[138]: type.StrictPointer.html
[139]: type.VisitWith.html
