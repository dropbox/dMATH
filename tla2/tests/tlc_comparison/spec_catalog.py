"""
TLA+ Examples Spec Catalog

Auto-generated catalog of all TLA+ example specifications.
Total: 176 specs across 70 categories.

Coverage breakdown (M audit 2026-01-13):
- 44 skipped: legitimately can't run (no INIT, animation, TLAPS deps)
- 1 TLA2 limitation: performance (MCInnerSerial - SUBSET enumeration timeout)
- 2 TLA2 P1 bugs (open issues):
  - AllocatorImplementation: BUG #87 (false positive invariant violation)
  - PaxosCommit: BUG #86 (state count mismatch - 19.5% over-exploration)
- Specs previously fixed (verified):
  - bosco: FIXED #56 (~1.1x TLC, was 3.2x)
  - FastPaxos: FIXED in W#94 (25617 states = TLC)
  - dijkstra-mutex_LSpec-model: FIXED in W#93 (90882 states = TLC)
  - MCCheckpointCoordination: FIXED (tla_library FiniteSets/Bags shadowing)
- ~130+ passing: verified against TLC baseline
"""

from typing import NamedTuple, Optional, List

class SpecInfo(NamedTuple):
    """Specification metadata."""
    name: str
    tla_path: str
    cfg_path: str
    category: str
    expected_states: Optional[int] = None  # TLC baseline state count
    timeout_seconds: int = 120
    notes: str = ""

# All discovered specifications
ALL_SPECS: List[SpecInfo] = [
    SpecInfo("MCBakery", "Bakery-Boulangerie/MCBakery.tla", "Bakery-Boulangerie/MCBakery.cfg", "Bakery_Boulangerie"),
    SpecInfo("MCBoulanger", "Bakery-Boulangerie/MCBoulanger.tla", "Bakery-Boulangerie/MCBoulanger.cfg", "Bakery_Boulangerie"),
    SpecInfo("CarTalkPuzzle_M1", "CarTalkPuzzle/CarTalkPuzzle.toolbox/Model_1/MC.tla", "CarTalkPuzzle/CarTalkPuzzle.toolbox/Model_1/MC.cfg", "CarTalkPuzzle"),
    SpecInfo("CarTalkPuzzle_M2", "CarTalkPuzzle/CarTalkPuzzle.toolbox/Model_2/MC.tla", "CarTalkPuzzle/CarTalkPuzzle.toolbox/Model_2/MC.cfg", "CarTalkPuzzle"),
    SpecInfo("CarTalkPuzzle_M3", "CarTalkPuzzle/CarTalkPuzzle.toolbox/Model_3/MC.tla", "CarTalkPuzzle/CarTalkPuzzle.toolbox/Model_3/MC.cfg", "CarTalkPuzzle"),
    SpecInfo("Chameneos", "Chameneos/Chameneos.tla", "Chameneos/Chameneos.cfg", "Chameneos"),
    SpecInfo("MCCheckpointCoordination", "CheckpointCoordination/MCCheckpointCoordination.tla", "CheckpointCoordination/MCCheckpointCoordination.cfg", "CheckpointCoordination"),
    SpecInfo("CigaretteSmokers", "CigaretteSmokers/CigaretteSmokers.tla", "CigaretteSmokers/CigaretteSmokers.cfg", "CigaretteSmokers"),
    SpecInfo("DieHard", "DieHard/DieHard.tla", "DieHard/DieHard.cfg", "DieHard"),
    SpecInfo("MCDieHarder", "DieHard/MCDieHarder.tla", "DieHard/MCDieHarder.cfg", "DieHard"),
    SpecInfo("DiningPhilosophers", "DiningPhilosophers/DiningPhilosophers.tla", "DiningPhilosophers/DiningPhilosophers.cfg", "DiningPhilosophers"),
    SpecInfo("Disruptor_MPMC", "Disruptor/Disruptor_MPMC.tla", "Disruptor/Disruptor_MPMC.cfg", "Disruptor"),
    SpecInfo("Disruptor_SPMC", "Disruptor/Disruptor_SPMC.tla", "Disruptor/Disruptor_SPMC.cfg", "Disruptor"),
    SpecInfo("Einstein", "EinsteinRiddle/Einstein.tla", "EinsteinRiddle/Einstein.cfg", "EinsteinRiddle"),
    SpecInfo("MCCRDT", "FiniteMonotonic/MCCRDT.tla", "FiniteMonotonic/MCCRDT.cfg", "FiniteMonotonic"),
    SpecInfo("MCDistributedReplicatedLog", "FiniteMonotonic/MCDistributedReplicatedLog.tla", "FiniteMonotonic/MCDistributedReplicatedLog.cfg", "FiniteMonotonic"),
    SpecInfo("MCReplicatedLog", "FiniteMonotonic/MCReplicatedLog.tla", "FiniteMonotonic/MCReplicatedLog.cfg", "FiniteMonotonic"),
    SpecInfo("GameOfLife", "GameOfLife/GameOfLife.tla", "GameOfLife/GameOfLife.cfg", "GameOfLife"),
    SpecInfo("Huang", "Huang/Huang.tla", "Huang/Huang.cfg", "Huang"),
    SpecInfo("MCKVsnap", "KeyValueStore/MCKVsnap.tla", "KeyValueStore/MCKVsnap.cfg", "KeyValueStore"),
    SpecInfo("SimKnuthYao", "KnuthYao/SimKnuthYao.tla", "KnuthYao/SimKnuthYao.cfg", "KnuthYao"),
    SpecInfo("MCFindHighest", "LearnProofs/MCFindHighest.tla", "LearnProofs/MCFindHighest.cfg", "LearnProofs"),
    SpecInfo("MCBinarySearch", "LoopInvariance/MCBinarySearch.tla", "LoopInvariance/MCBinarySearch.cfg", "LoopInvariance"),
    SpecInfo("MCQuicksort", "LoopInvariance/MCQuicksort.tla", "LoopInvariance/MCQuicksort.cfg", "LoopInvariance"),
    SpecInfo("MCMajority", "Majority/MCMajority.tla", "Majority/MCMajority.cfg", "Majority"),
    SpecInfo("MCParReach", "MisraReachability/MCParReach.tla", "MisraReachability/MCParReach.cfg", "MisraReachability"),
    SpecInfo("MCReachable", "MisraReachability/MCReachable.tla", "MisraReachability/MCReachable.cfg", "MisraReachability"),
    SpecInfo("MissionariesAndCannibals", "MissionariesAndCannibals/MissionariesAndCannibals.tla", "MissionariesAndCannibals/MissionariesAndCannibals.cfg", "MissionariesAndCannibals"),
    SpecInfo("MultiPaxos_MC", "MultiPaxos-SMR/MultiPaxos_MC.tla", "MultiPaxos-SMR/MultiPaxos_MC.cfg", "MultiPaxos_SMR"),
    SpecInfo("N-Queens_FourQueens", "N-Queens/Queens.toolbox/FourQueens/MC.tla", "N-Queens/Queens.toolbox/FourQueens/MC.cfg", "N_Queens"),
    SpecInfo("N-Queens_FourQueens_2", "N-Queens/QueensPluscal.toolbox/FourQueens/MC.tla", "N-Queens/QueensPluscal.toolbox/FourQueens/MC.cfg", "N_Queens"),
    SpecInfo("MCConsensus", "Paxos/MCConsensus.tla", "Paxos/MCConsensus.cfg", "Paxos"),
    SpecInfo("MCPaxos", "Paxos/MCPaxos.tla", "Paxos/MCPaxos.cfg", "Paxos"),
    SpecInfo("MCVoting", "Paxos/MCVoting.tla", "Paxos/MCVoting.cfg", "Paxos"),
    SpecInfo("MCConsensus_2", "PaxosHowToWinATuringAward/MCConsensus.tla", "PaxosHowToWinATuringAward/MCConsensus.cfg", "PaxosHowToWinATuringAward"),
    SpecInfo("MCVoting_2", "PaxosHowToWinATuringAward/MCVoting.tla", "PaxosHowToWinATuringAward/MCVoting.cfg", "PaxosHowToWinATuringAward"),
    SpecInfo("Prisoners", "Prisoners/Prisoners.tla", "Prisoners/Prisoners.cfg", "Prisoners"),
    SpecInfo("Prisoner", "Prisoners_Single_Switch/Prisoner.tla", "Prisoners_Single_Switch/Prisoner.cfg", "Prisoners_Single_Switch"),
    SpecInfo("MC", "ReadersWriters/MC.tla", "ReadersWriters/MC.cfg", "ReadersWriters"),
    SpecInfo("MC_2", "SDP_Verification/SDP_Attack_New_Solution_Spec/MC.tla", "SDP_Verification/SDP_Attack_New_Solution_Spec/MC.cfg", "SDP_Verification"),
    SpecInfo("MC_3", "SDP_Verification/SDP_Attack_Spec/MC.tla", "SDP_Verification/SDP_Attack_Spec/MC.cfg", "SDP_Verification"),
    SpecInfo("FastPaxos", "SimplifiedFastPaxos/FastPaxos.tla", "SimplifiedFastPaxos/FastPaxos.cfg", "SimplifiedFastPaxos"),
    SpecInfo("Paxos", "SimplifiedFastPaxos/Paxos.tla", "SimplifiedFastPaxos/Paxos.cfg", "SimplifiedFastPaxos"),
    SpecInfo("MC_4", "SingleLaneBridge/MC.tla", "SingleLaneBridge/MC.cfg", "SingleLaneBridge"),
    SpecInfo("SlidingPuzzles", "SlidingPuzzles/SlidingPuzzles.tla", "SlidingPuzzles/SlidingPuzzles.cfg", "SlidingPuzzles"),
    SpecInfo("SpanTree", "SpanningTree/SpanTree.tla", "SpanningTree/SpanTree.cfg", "SpanningTree"),
    SpecInfo("SpanTreeRandom", "SpanningTree/SpanTreeRandom.tla", "SpanningTree/SpanTreeRandom.cfg", "SpanningTree"),
    SpecInfo("MCInnerSequential", "SpecifyingSystems/AdvancedExamples/MCInnerSequential.tla", "SpecifyingSystems/AdvancedExamples/MCInnerSequential.cfg", "SpecifyingSystems"),
    SpecInfo("MCInnerSerial", "SpecifyingSystems/AdvancedExamples/MCInnerSerial.tla", "SpecifyingSystems/AdvancedExamples/MCInnerSerial.cfg", "SpecifyingSystems"),
    SpecInfo("AsynchInterface", "SpecifyingSystems/AsynchronousInterface/AsynchInterface.tla", "SpecifyingSystems/AsynchronousInterface/AsynchInterface.cfg", "SpecifyingSystems"),
    SpecInfo("Channel", "SpecifyingSystems/AsynchronousInterface/Channel.tla", "SpecifyingSystems/AsynchronousInterface/Channel.cfg", "SpecifyingSystems"),
    SpecInfo("PrintValues", "SpecifyingSystems/AsynchronousInterface/PrintValues.tla", "SpecifyingSystems/AsynchronousInterface/PrintValues.cfg", "SpecifyingSystems"),
    SpecInfo("MCInternalMemory", "SpecifyingSystems/CachingMemory/MCInternalMemory.tla", "SpecifyingSystems/CachingMemory/MCInternalMemory.cfg", "SpecifyingSystems"),
    SpecInfo("MCWriteThroughCache", "SpecifyingSystems/CachingMemory/MCWriteThroughCache.tla", "SpecifyingSystems/CachingMemory/MCWriteThroughCache.cfg", "SpecifyingSystems"),
    SpecInfo("MCInnerFIFO", "SpecifyingSystems/FIFO/MCInnerFIFO.tla", "SpecifyingSystems/FIFO/MCInnerFIFO.cfg", "SpecifyingSystems"),
    SpecInfo("HourClock", "SpecifyingSystems/HourClock/HourClock.tla", "SpecifyingSystems/HourClock/HourClock.cfg", "SpecifyingSystems"),
    SpecInfo("HourClock2", "SpecifyingSystems/HourClock/HourClock2.tla", "SpecifyingSystems/HourClock/HourClock2.cfg", "SpecifyingSystems"),
    SpecInfo("LiveHourClock", "SpecifyingSystems/Liveness/LiveHourClock.tla", "SpecifyingSystems/Liveness/LiveHourClock.cfg", "SpecifyingSystems"),
    SpecInfo("MCLiveInternalMemory", "SpecifyingSystems/Liveness/MCLiveInternalMemory.tla", "SpecifyingSystems/Liveness/MCLiveInternalMemory.cfg", "SpecifyingSystems"),
    SpecInfo("MCLiveWriteThroughCache", "SpecifyingSystems/Liveness/MCLiveWriteThroughCache.tla", "SpecifyingSystems/Liveness/MCLiveWriteThroughCache.cfg", "SpecifyingSystems"),
    SpecInfo("MCRealTimeHourClock", "SpecifyingSystems/RealTime/MCRealTimeHourClock.tla", "SpecifyingSystems/RealTime/MCRealTimeHourClock.cfg", "SpecifyingSystems"),
    SpecInfo("SimpleMath", "SpecifyingSystems/SimpleMath/SimpleMath.tla", "SpecifyingSystems/SimpleMath/SimpleMath.cfg", "SpecifyingSystems"),
    SpecInfo("ABCorrectness", "SpecifyingSystems/TLC/ABCorrectness.tla", "SpecifyingSystems/TLC/ABCorrectness.cfg", "SpecifyingSystems"),
    SpecInfo("MCAlternatingBit", "SpecifyingSystems/TLC/MCAlternatingBit.tla", "SpecifyingSystems/TLC/MCAlternatingBit.cfg", "SpecifyingSystems"),
    SpecInfo("Stones", "Stones/Stones.tla", "Stones/Stones.cfg", "Stones"),
    SpecInfo("TestGraphs", "TLC/TestGraphs.tla", "TLC/TestGraphs.cfg", "TLC"),
    SpecInfo("Simple", "TeachingConcurrency/Simple.tla", "TeachingConcurrency/Simple.cfg", "TeachingConcurrency"),
    SpecInfo("SimpleRegular", "TeachingConcurrency/SimpleRegular.tla", "TeachingConcurrency/SimpleRegular.cfg", "TeachingConcurrency"),
    SpecInfo("TransitiveClosure", "TransitiveClosure/TransitiveClosure.tla", "TransitiveClosure/TransitiveClosure.cfg", "TransitiveClosure"),
    SpecInfo("MCTwoPhase", "TwoPhase/MCTwoPhase.tla", "TwoPhase/MCTwoPhase.cfg", "TwoPhase"),
    SpecInfo("MCYoYoNoPruning", "YoYo/MCYoYoNoPruning.tla", "YoYo/MCYoYoNoPruning.cfg", "YoYo"),
    SpecInfo("MCYoYoPruning", "YoYo/MCYoYoPruning.tla", "YoYo/MCYoYoPruning.cfg", "YoYo"),
    SpecInfo("YoYoAllGraphs", "YoYo/YoYoAllGraphs.tla", "YoYo/YoYoAllGraphs.cfg", "YoYo"),
    SpecInfo("aba_asyn_byz", "aba-asyn-byz/aba_asyn_byz.tla", "aba-asyn-byz/aba_asyn_byz.cfg", "aba_asyn_byz"),
    SpecInfo("ACP_NB_TLC", "acp/ACP_NB_TLC.tla", "acp/ACP_NB_TLC.cfg", "acp"),
    SpecInfo("ACP_NB_WRONG_TLC", "acp/ACP_NB_WRONG_TLC.tla", "acp/ACP_NB_WRONG_TLC.cfg", "acp"),
    SpecInfo("ACP_SB_TLC", "acp/ACP_SB_TLC.tla", "acp/ACP_SB_TLC.cfg", "acp"),
    SpecInfo("AllocatorImplementation", "allocator/AllocatorImplementation.tla", "allocator/AllocatorImplementation.cfg", "allocator"),
    SpecInfo("AllocatorRefinement", "allocator/AllocatorRefinement.tla", "allocator/AllocatorRefinement.cfg", "allocator"),
    SpecInfo("SchedulingAllocator", "allocator/SchedulingAllocator.tla", "allocator/SchedulingAllocator.cfg", "allocator"),
    SpecInfo("SimpleAllocator", "allocator/SimpleAllocator.tla", "allocator/SimpleAllocator.cfg", "allocator"),
    SpecInfo("Barrier", "barriers/Barrier.tla", "barriers/Barrier.cfg", "barriers"),
    SpecInfo("Barriers", "barriers/Barriers.tla", "barriers/Barriers.cfg", "barriers"),
    SpecInfo("bcastByz", "bcastByz/bcastByz.tla", "bcastByz/bcastByz.cfg", "bcastByz"),
    SpecInfo("bcastFolklore", "bcastFolklore/bcastFolklore.tla", "bcastFolklore/bcastFolklore.cfg", "bcastFolklore"),
    SpecInfo("bosco", "bosco/bosco.tla", "bosco/bosco.cfg", "bosco"),
    SpecInfo("BufferedRandomAccessFile", "braf/BufferedRandomAccessFile.tla", "braf/BufferedRandomAccessFile.cfg", "braf"),
    SpecInfo("btree", "btree/btree.tla", "btree/btree.cfg", "btree"),
    SpecInfo("kvstore", "btree/kvstore.tla", "btree/kvstore.cfg", "btree"),
    SpecInfo("VoucherCancel", "byihive/VoucherCancel.tla", "byihive/VoucherCancel.cfg", "byihive"),
    SpecInfo("VoucherIssue", "byihive/VoucherIssue.tla", "byihive/VoucherIssue.cfg", "byihive"),
    SpecInfo("VoucherLifeCycle", "byihive/VoucherLifeCycle.tla", "byihive/VoucherLifeCycle.cfg", "byihive"),
    SpecInfo("VoucherRedeem", "byihive/VoucherRedeem.tla", "byihive/VoucherRedeem.cfg", "byihive"),
    SpecInfo("VoucherTransfer", "byihive/VoucherTransfer.tla", "byihive/VoucherTransfer.cfg", "byihive"),
    SpecInfo("BPConProof", "byzpaxos/BPConProof.tla", "byzpaxos/BPConProof.cfg", "byzpaxos"),
    SpecInfo("Consensus", "byzpaxos/Consensus.tla", "byzpaxos/Consensus.cfg", "byzpaxos"),
    SpecInfo("PConProof", "byzpaxos/PConProof.tla", "byzpaxos/PConProof.cfg", "byzpaxos"),
    SpecInfo("VoteProof", "byzpaxos/VoteProof.tla", "byzpaxos/VoteProof.cfg", "byzpaxos"),
    SpecInfo("c1cs", "c1cs/c1cs.tla", "c1cs/c1cs.cfg", "c1cs"),
    SpecInfo("cf1s_folklore", "cf1s-folklore/cf1s_folklore.tla", "cf1s-folklore/cf1s_folklore.cfg", "cf1s_folklore"),
    SpecInfo("MCChangRoberts", "chang_roberts/MCChangRoberts.tla", "chang_roberts/MCChangRoberts.cfg", "chang_roberts"),
    SpecInfo("EnvironmentController", "detector_chan96/EnvironmentController.tla", "detector_chan96/EnvironmentController.cfg", "detector_chan96"),
    SpecInfo("dijkstra-mutex_LSpec-model", "dijkstra-mutex/DijkstraMutex.toolbox/LSpec-model/MC.tla", "dijkstra-mutex/DijkstraMutex.toolbox/LSpec-model/MC.cfg", "dijkstra_mutex"),
    SpecInfo("dijkstra-mutex_Safety-4-processors", "dijkstra-mutex/DijkstraMutex.toolbox/Safety-4-processors/MC.tla", "dijkstra-mutex/DijkstraMutex.toolbox/Safety-4-processors/MC.cfg", "dijkstra_mutex"),
    SpecInfo("MC_HDiskSynod", "diskpaxos/MC_HDiskSynod.tla", "diskpaxos/MC_HDiskSynod.cfg", "diskpaxos"),
    SpecInfo("MCEcho", "echo/MCEcho.tla", "echo/MCEcho.cfg", "echo"),
    SpecInfo("SimTokenRing", "ewd426/SimTokenRing.tla", "ewd426/SimTokenRing.cfg", "ewd426"),
    SpecInfo("TokenRing", "ewd426/TokenRing.tla", "ewd426/TokenRing.cfg", "ewd426"),
    SpecInfo("EWD687a_anim", "ewd687a/EWD687a_anim.tla", "ewd687a/EWD687a_anim.cfg", "ewd687a"),
    SpecInfo("MCEWD687a", "ewd687a/MCEWD687a.tla", "ewd687a/MCEWD687a.cfg", "ewd687a"),
    SpecInfo("EWD840", "ewd840/EWD840.tla", "ewd840/EWD840.cfg", "ewd840"),
    SpecInfo("EWD840_anim", "ewd840/EWD840_anim.tla", "ewd840/EWD840_anim.cfg", "ewd840"),
    SpecInfo("EWD840_json", "ewd840/EWD840_json.tla", "ewd840/EWD840_json.cfg", "ewd840"),
    SpecInfo("SyncTerminationDetection", "ewd840/SyncTerminationDetection.tla", "ewd840/SyncTerminationDetection.cfg", "ewd840"),
    SpecInfo("AsyncTerminationDetection", "ewd998/AsyncTerminationDetection.tla", "ewd998/AsyncTerminationDetection.cfg", "ewd998"),
    SpecInfo("EWD998", "ewd998/EWD998.tla", "ewd998/EWD998.cfg", "ewd998"),
    SpecInfo("EWD998Chan", "ewd998/EWD998Chan.tla", "ewd998/EWD998Chan.cfg", "ewd998"),
    SpecInfo("EWD998ChanID", "ewd998/EWD998ChanID.tla", "ewd998/EWD998ChanID.cfg", "ewd998"),
    SpecInfo("EWD998ChanID_export", "ewd998/EWD998ChanID_export.tla", "ewd998/EWD998ChanID_export.cfg", "ewd998"),
    SpecInfo("EWD998ChanID_shiviz", "ewd998/EWD998ChanID_shiviz.tla", "ewd998/EWD998ChanID_shiviz.cfg", "ewd998"),
    SpecInfo("EWD998ChanTrace", "ewd998/EWD998ChanTrace.tla", "ewd998/EWD998ChanTrace.cfg", "ewd998"),
    SpecInfo("EWD998PCal", "ewd998/EWD998PCal.tla", "ewd998/EWD998PCal.cfg", "ewd998"),
    SpecInfo("SmokeEWD998", "ewd998/SmokeEWD998.tla", "ewd998/SmokeEWD998.cfg", "ewd998"),
    SpecInfo("SmokeEWD998_SC", "ewd998/SmokeEWD998_SC.tla", "ewd998/SmokeEWD998_SC.cfg", "ewd998"),
    SpecInfo("clean", "glowingRaccoon/clean.tla", "glowingRaccoon/clean.cfg", "glowingRaccoon"),
    SpecInfo("product", "glowingRaccoon/product.tla", "glowingRaccoon/product.cfg", "glowingRaccoon"),
    SpecInfo("stages", "glowingRaccoon/stages.tla", "glowingRaccoon/stages.cfg", "glowingRaccoon"),
    SpecInfo("MCLamportMutex", "lamport_mutex/MCLamportMutex.tla", "lamport_mutex/MCLamportMutex.cfg", "lamport_mutex"),
    SpecInfo("Lock", "locks_auxiliary_vars/Lock.tla", "locks_auxiliary_vars/Lock.cfg", "locks_auxiliary_vars"),
    SpecInfo("LockHS", "locks_auxiliary_vars/LockHS.tla", "locks_auxiliary_vars/LockHS.cfg", "locks_auxiliary_vars"),
    SpecInfo("Peterson", "locks_auxiliary_vars/Peterson.tla", "locks_auxiliary_vars/Peterson.cfg", "locks_auxiliary_vars"),
    SpecInfo("nbacc_ray97", "nbacc_ray97/nbacc_ray97.tla", "nbacc_ray97/nbacc_ray97.cfg", "nbacc_ray97"),
    SpecInfo("nbacg_guer01", "nbacg_guer01/nbacg_guer01.tla", "nbacg_guer01/nbacg_guer01.cfg", "nbacg_guer01"),
    SpecInfo("MC_spanning", "spanning/MC_spanning.tla", "spanning/MC_spanning.cfg", "spanning"),
    SpecInfo("MC_sums_even", "sums_even/MC_sums_even.tla", "sums_even/MC_sums_even.cfg", "sums_even"),
    SpecInfo("MCtcp", "tcp/MCtcp.tla", "tcp/MCtcp.cfg", "tcp"),
    SpecInfo("tower_of_hanoi_M1", "tower_of_hanoi/Hanoi.toolbox/Model_1/MC.tla", "tower_of_hanoi/Hanoi.toolbox/Model_1/MC.cfg", "tower_of_hanoi"),
    SpecInfo("2PCwithBTM", "transaction_commit/2PCwithBTM.tla", "transaction_commit/2PCwithBTM.cfg", "transaction_commit"),
    SpecInfo("PaxosCommit", "transaction_commit/PaxosCommit.tla", "transaction_commit/PaxosCommit.cfg", "transaction_commit"),
    SpecInfo("TCommit", "transaction_commit/TCommit.tla", "transaction_commit/TCommit.cfg", "transaction_commit"),
    SpecInfo("TwoPhase", "transaction_commit/TwoPhase.tla", "transaction_commit/TwoPhase.cfg", "transaction_commit"),
    # === Added for 100% coverage (iteration #1) ===
    SpecInfo("MCCheckpointCoordinationFailure", "CheckpointCoordination/MCCheckpointCoordination.tla", "CheckpointCoordination/MCCheckpointCoordinationFailure.cfg", "CheckpointCoordination"),
    SpecInfo("CoffeeCan1000Beans", "CoffeeCan/CoffeeCan.tla", "CoffeeCan/CoffeeCan1000Beans.cfg", "CoffeeCan"),
    SpecInfo("CoffeeCan100Beans", "CoffeeCan/CoffeeCan.tla", "CoffeeCan/CoffeeCan100Beans.cfg", "CoffeeCan"),
    SpecInfo("CoffeeCan3000Beans", "CoffeeCan/CoffeeCan.tla", "CoffeeCan/CoffeeCan3000Beans.cfg", "CoffeeCan"),
    SpecInfo("Disruptor_MPMC_liveliness", "Disruptor/Disruptor_MPMC.tla", "Disruptor/Disruptor_MPMC_liveliness.cfg", "Disruptor"),
    SpecInfo("MCKVSSafetyLarge", "KeyValueStore/MCKVS.tla", "KeyValueStore/MCKVSSafetyLarge.cfg", "KeyValueStore"),
    SpecInfo("MCKVSSafetyMedium", "KeyValueStore/MCKVS.tla", "KeyValueStore/MCKVSSafetyMedium.cfg", "KeyValueStore"),
    SpecInfo("MCKVSSafetySmall", "KeyValueStore/MCKVS.tla", "KeyValueStore/MCKVSSafetySmall.cfg", "KeyValueStore"),
    SpecInfo("MCLeastCircularSubstringMedium", "LeastCircularSubstring/MCLeastCircularSubstring.tla", "LeastCircularSubstring/MCLeastCircularSubstringMedium.cfg", "LeastCircularSubstring"),
    SpecInfo("MCLeastCircularSubstringSmall", "LeastCircularSubstring/MCLeastCircularSubstring.tla", "LeastCircularSubstring/MCLeastCircularSubstringSmall.cfg", "LeastCircularSubstring"),
    SpecInfo("MCReachabilityTestAllGraphs", "MisraReachability/ReachabilityTest.tla", "MisraReachability/MCReachabilityTestAllGraphs.cfg", "MisraReachability"),
    SpecInfo("MCReachabilityTestRandomGraphs", "MisraReachability/ReachabilityTest.tla", "MisraReachability/MCReachabilityTestRandomGraphs.cfg", "MisraReachability"),
    SpecInfo("CatEvenBoxes", "Moving_Cat_Puzzle/Cat.tla", "Moving_Cat_Puzzle/CatEvenBoxes.cfg", "Moving_Cat_Puzzle"),
    SpecInfo("CatOddBoxes", "Moving_Cat_Puzzle/Cat.tla", "Moving_Cat_Puzzle/CatOddBoxes.cfg", "Moving_Cat_Puzzle"),
    SpecInfo("ElevatorLivenessLarge", "MultiCarElevator/Elevator.tla", "MultiCarElevator/ElevatorLivenessLarge.cfg", "MultiCarElevator"),
    SpecInfo("ElevatorLivenessMedium", "MultiCarElevator/Elevator.tla", "MultiCarElevator/ElevatorLivenessMedium.cfg", "MultiCarElevator"),
    SpecInfo("ElevatorSafetyLarge", "MultiCarElevator/Elevator.tla", "MultiCarElevator/ElevatorSafetyLarge.cfg", "MultiCarElevator"),
    SpecInfo("ElevatorSafetyMedium", "MultiCarElevator/Elevator.tla", "MultiCarElevator/ElevatorSafetyMedium.cfg", "MultiCarElevator"),
    SpecInfo("ElevatorSafetySmall", "MultiCarElevator/Elevator.tla", "MultiCarElevator/ElevatorSafetySmall.cfg", "MultiCarElevator"),
    SpecInfo("MultiPaxos_MC_small", "MultiPaxos-SMR/MultiPaxos_MC.tla", "MultiPaxos-SMR/MultiPaxos_MC_small.cfg", "MultiPaxos_SMR"),
    SpecInfo("MCNanoLarge", "NanoBlockchain/Nano.tla", "NanoBlockchain/MCNanoLarge.cfg", "NanoBlockchain"),
    SpecInfo("MCNanoMedium", "NanoBlockchain/Nano.tla", "NanoBlockchain/MCNanoMedium.cfg", "NanoBlockchain"),
    SpecInfo("MCNanoSmall", "NanoBlockchain/Nano.tla", "NanoBlockchain/MCNanoSmall.cfg", "NanoBlockchain"),
    SpecInfo("MCPaxosSmall", "PaxosHowToWinATuringAward/Paxos.tla", "PaxosHowToWinATuringAward/MCPaxosSmall.cfg", "PaxosHowToWinATuringAward"),
    SpecInfo("MCPaxosTiny", "PaxosHowToWinATuringAward/Paxos.tla", "PaxosHowToWinATuringAward/MCPaxosTiny.cfg", "PaxosHowToWinATuringAward"),
    SpecInfo("PrisonerLightUnknown", "Prisoners_Single_Switch/Prisoner.tla", "Prisoners_Single_Switch/PrisonerLightUnknown.cfg", "Prisoners_Single_Switch"),
    SpecInfo("PrisonerSolo", "Prisoners_Single_Switch/Prisoner.tla", "Prisoners_Single_Switch/PrisonerSolo.cfg", "Prisoners_Single_Switch"),
    SpecInfo("PrisonerSoloLightUnknown", "Prisoners_Single_Switch/Prisoner.tla", "Prisoners_Single_Switch/PrisonerSoloLightUnknown.cfg", "Prisoners_Single_Switch"),
    SpecInfo("SlushLarge", "SlushProtocol/Slush.tla", "SlushProtocol/SlushLarge.cfg", "SlushProtocol"),
    SpecInfo("SlushMedium", "SlushProtocol/Slush.tla", "SlushProtocol/SlushMedium.cfg", "SlushProtocol"),
    SpecInfo("SlushSmall", "SlushProtocol/Slush.tla", "SlushProtocol/SlushSmall.cfg", "SlushProtocol"),
    SpecInfo("SpanTreeTest4Nodes", "SpanningTree/SpanTree.tla", "SpanningTree/SpanTreeTest4Nodes.cfg", "SpanningTree"),
    SpecInfo("SpanTreeTest5Nodes", "SpanningTree/SpanTree.tla", "SpanningTree/SpanTreeTest5Nodes.cfg", "SpanningTree"),
    SpecInfo("bcastByzNoBcast", "bcastByz/bcastByz.tla", "bcastByz/bcastByzNoBcast.cfg", "bcastByz"),
    SpecInfo("EWD998Small", "ewd998/EWD998.tla", "ewd998/EWD998Small.cfg", "ewd998"),
]

# Known specs requiring special handling
SKIP_SPECS = {
    # Apalache-specific (TLC can't run these either)
    "Einstein": "Requires Apalache module (TLC fails too)",
    # Simulation mode specs
    "SimKnuthYao": "Requires simulation mode",
    "SimTokenRing": "Requires simulation mode",
    # Animation/export specs
    "EWD840_anim": "Animation spec",
    "EWD687a_anim": "Animation spec",
    "EWD840_json": "JSON export spec",
    "EWD998ChanID_export": "Export spec",
    "EWD998ChanID_shiviz": "ShiViz export spec",
    "EWD998ChanTrace": "Trace export spec",
    # Constant evaluation specs (no INIT/NEXT, only ASSUME checks)
    # TLA2 doesn't support constant evaluation mode - these need INIT/NEXT
    "CarTalkPuzzle_M1": "Constant evaluation (no INIT)",
    "CarTalkPuzzle_M2": "Constant evaluation (no INIT)",
    "CarTalkPuzzle_M3": "Constant evaluation (no INIT)",
    "SimpleMath": "Constant evaluation (no INIT)",
    "Stones": "Constant evaluation (no INIT)",
    "TransitiveClosure": "Constant evaluation (no INIT)",
    "MC_sums_even": "Constant evaluation (no INIT)",
    "SmokeEWD998": "Constant evaluation (no INIT)",
    "SmokeEWD998_SC": "Constant evaluation (no INIT)",
    "PrintValues": "Print spec (no INIT)",
    "TestGraphs": "TLC test spec",
    # Problematic specs
    "SpanTreeRandom": "Uses RandomElement (not supported)",
    "YoYoAllGraphs": "Very slow enumeration",
    # NOTE: TLAPS modules (TLAPS.tla, FiniteSetTheorems.tla, NaturalsInduction.tla,
    # FunctionsFork.tla, etc.) are now available in test_specs/tla_library/.
    # Many previously skipped specs now work. Remaining skips below have other issues.
    # NOTE: Consensus and VoteProof DO have config files and work correctly.
    # Removed from skip list after fixing FiniteSetTheorems.tla (Fixes #108).
    # TLA2 Init enumeration bugs (not TLAPS-related)
    "LockHS": "TLA2 bug: Init enumeration unsupported expressions (Stuttering module)",
    "bcastByz": "TLA2 bug: Init enumeration finds no solutions (TLC finds 64)",
    "bcastByzNoBcast": "TLA2 bug: same as bcastByz",
    # TLA2 primed variable evaluation bug
    "MCTwoPhase": "TLA2 bug: Primed variable in non-next-state context",
}

# Known large specs (>100K states, need extended timeout)
# Value is subprocess timeout in seconds; pytest-timeout gets 2x+60 buffer
LARGE_SPECS = {
    "TokenRing": 300,
    "Huang": 300,
    "MCBakery": 600,
    "MCBoulanger": 600,
    "bcastFolklore": 600,
    "CoffeeCan1000Beans": 300,
    "CoffeeCan3000Beans": 600,
    # Added based on timeout failures - need investigation
    "MCCheckpointCoordination": 700,  # TLA2 completes in ~600s, 4x slower than TLC (~150s)
    "Disruptor_MPMC": 300,  # TLA2 timed out at 120s
    "Disruptor_SPMC": 300,  # TLA2 timed out at 120s
    # Additional slow specs identified in iteration 452
    "btree": 300,  # TLA2 times out at 120s
    "c1cs": 300,  # Slow enumeration
    "cf1s_folklore": 300,  # Slow liveness check
    "EWD998": 300,  # Slow
    "EWD998Chan": 300,  # Slow
    "EWD998ChanID": 300,  # Slow
    "EWD998PCal": 300,  # Slow
    "MCLamportMutex": 300,  # Slow
    "FastPaxos": 300,  # 25617 states, ~70s TLA2 runtime
    "MCReachable": 300,  # Slow enumeration
    "GameOfLife": 300,  # 65536 states
    "Chameneos": 300,  # 34534 states
    "MCKVsnap": 300,  # 32293 states
    "nbacg_guer01": 300,  # 24922 states
    "Barriers": 300,  # 29279 states
    "MC": 300,  # ReadersWriters - 21527 states
    "PaxosCommit": 600,  # 1.3M states, ~360s on test machine
    "aba_asyn_byz": 300,  # Slow
    "MCInnerSerial": 300,  # Slow
    "MCEWD687a": 300,  # Slow
    "EnvironmentController": 300,  # Slow
    "MCtcp": 300,  # Slow (few states but complex)
    "MC_HDiskSynod": 600,  # Slow: TLC also exceeds 120s default timeout
    "bosco": 600,  # Large spec - ~6:49 runtime (~1.1x TLC). FIXED #56 (was 3.2x)
    "MultiPaxos_MC": 600,  # Large spec (FIXED in #458)
    "dijkstra-mutex_Safety-4-processors": 600,  # 535K+ states with 4 processes
    "BufferedRandomAccessFile": 300,  # 3316 states but complex enumeration (~160s)
    # Slush protocol specs - 274K states, TLA2 is slower than TLC
    "SlushSmall": 300,  # 274678 states, ~147s TLA2 vs ~1s TLC
    "SlushMedium": 600,  # Medium config, extended timeout
    "SlushLarge": 900,  # Large config, extended timeout
    # KeyValueStore safety specs - large state space
    "MCKVSSafetyLarge": 600,  # TLC timed out at 120s
    # EWD998 specs - termination detection with message passing
    "EWD998Small": 300,  # EWD998 with smaller parameters, still needs timeout
    # Added in W#3 - specs that timed out at 120s
    # Elevator specs (MultiCarElevator)
    "ElevatorLivenessLarge": 600,  # Liveness check on large elevator config
    "ElevatorLivenessMedium": 300,  # Liveness check on medium elevator config
    "ElevatorSafetyLarge": 600,  # Safety check on large elevator config
    "ElevatorSafetyMedium": 300,  # Safety check on medium elevator config
    "ElevatorSafetySmall": 300,  # Safety check on small elevator config
    # KeyValueStore safety specs
    "MCKVSSafetyMedium": 300,  # KVS safety medium config
    "MCKVSSafetySmall": 300,  # KVS safety small config
    # LeastCircularSubstring specs
    "MCLeastCircularSubstringMedium": 300,  # Medium config
    "MCLeastCircularSubstringSmall": 300,  # Small config
    # MultiPaxos small config
    "MultiPaxos_MC_small": 300,  # MultiPaxos smaller config
}

# Specs with known TLC errors (not TLA2 bugs)
TLC_ISSUES = {
    # None known yet
}

# Specs with known TLA2 limitations (not bugs, just limitations)
# AUDIT 2026-01-05: Cleaned stale entries, fixed incorrect skip reasons
TLA2_LIMITATIONS = {
    # Init enumeration limitations
    # FastPaxos: REMOVED - now works (25617 states, verified 2026-01-05)
    # MC_2: REMOVED - Bitwise module now implemented (2026-01-06)
    # MC_3: REMOVED - Bitwise module now implemented (2026-01-06)
    # Prisoner: REMOVED - negation in Init now handled (2026-01-06, 16 states, verified)
    # MCInnerFIFO: REMOVED - in bash tests (3864 states)
    # MCYoYoNoPruning: REMOVED - now works (60 states, verified 2026-01-05)
    # MCYoYoPruning: REMOVED - now works (102 states, in bash tests)
    # AllocatorImplementation: FIXED in #557 - temporal identifier in PROPERTY decomposition
    #   was incorrectly categorized as init term (Liveness with WF/SF). Fixed by checking
    #   if referenced operator contains temporal operators before categorization.
    # Performance limitations (not correctness bugs)
    "MCInnerSerial": "Performance: eager SUBSET enumeration times out (TLC has optimizations)",
    # MCCheckpointCoordination: MOVED to TLA2_BUGS - actually a correctness bug (see #13)
    # bosco: REMOVED from limitations - now ~1.1x TLC (FIXED #56), close to <1x target
}

# Specs needing investigation (potential TLA2 bugs)
# M#57 audit (2026-01-11): Found 19 test failures falsely claimed fixed
# M#61 audit (2026-01-11): Cat specs verified FIXED (f7e6163), 17 remain
# W#?: ENABLED eval fix (#55) - Fixed 8 liveness false positives, 9 remain
# W#45 (2026-01-11): Config operator replacement + caching fix resolved 5 more
# See #54, #55 for details
TLA2_BUGS = {
    # === FALSE POSITIVE LIVENESS VIOLATIONS ===
    # TLA2 reports liveness error when TLC says property holds
    #
    # ENABLED evaluation fix (#55) resolved 8 specs:
    # - Prisoners: FIXED - fresh enumeration for ENABLED, not pre-computed successors
    # - MCInnerSequential: FIXED - same root cause
    # - MCAlternatingBit: FIXED - same root cause
    # - ACP_NB_TLC: FIXED - same root cause
    # - SchedulingAllocator: FIXED - same root cause
    # - SimpleAllocator: FIXED - same root cause
    # - VoucherIssue: FIXED - same root cause
    # - Disruptor_MPMC_liveliness: FIXED - same root cause
    #
    # Remaining issues (different root cause, likely stuttering cycle detection):
    # MC_4: FIXED - resolved by ENABLED evaluation fix (#55), verified passing
    # dijkstra-mutex_LSpec-model: FIXED in W#93 (b2b2179) - stuttering edge fix. 90882 states, matches TLC.
    # MCYoYoPruning: FIXED in W#45 (14da02d) - config operator replacement fix. 102=102 states.
    # AllocatorRefinement: FIXED in W#45 (14da02d) - same fix. 1690=1690 states.
    # CatEvenBoxes: FIXED in f7e6163 (stuttering edge fix). Verified 48 states, matches TLC.
    # CatOddBoxes: FIXED in f7e6163 (stuttering edge fix). Verified 30 states, matches TLC.

    # === STATE COUNT MISMATCHES (ALL FIXED) ===
    # TLA2 finds different number of states than TLC
    # M#73 audit: Paxos is FIXED (1207 states matches TLC).
    # W#94 audit (2026-01-11): FastPaxos FIXED (25617 states = TLC), PaxosCommit FIXED (1321761 states = TLC)
    # FastPaxos: FIXED in W#94 - 25617 states matches TLC exactly
    # PaxosCommit: FIXED in W#94 - 1321761 states matches TLC exactly
    # MCWriteThroughCache: FIXED in W#45 (14da02d) - caching fix. 5196=5196 states.
    # MCLiveInternalMemory: FIXED in W#45 (14da02d) - same fix. 4408=4408 states.
    # MCLiveWriteThroughCache: FIXED in W#45 (14da02d) - same fix. 5196=5196 states.
    # AllocatorImplementation: FIXED in #54/6aa041b - 17701 states, no errors (matches TLC)

    # === HISTORICAL (commented for reference) ===
    # MCAlternatingBit: claimed FIXED in iteration 445 (WF_/SF_ parsing fix)
    # MCRealTimeHourClock: FIXED in 04156f5 (M#7). Conflict detection for primed assignments +
    #   preprocessing disabled for liveness specs. TLA2 now finds 216 states and detects violation.
    # Cat specs: FIXED in W#5 - verified both CatEvenBoxes (48 states) and CatOddBoxes (30 states)
    # match TLC exactly. Both TLA2 and TLC report "No errors found" and state counts match.

    # === ALL LIVENESS BUGS FIXED (W session 2026-01-11) ===
    # Prisoner and PrisonerLightUnknown: FIXED by topological sort of symbolic assignments (7ad2079)
    # MCInnerSequential, MCLiveInternalMemory, MCLiveWriteThroughCache: FIXED
    #   (config-overridden constants now take precedence over operator definitions in liveness)
    # ACP_NB_TLC: FIXED - tuple subscript parsing in fairness constraints
    #   (was creating Expr::Ident("<<coordinator, participant>>") instead of Expr::Tuple)

    # === HISTORICAL FIXES (verified passing) ===
    # TLA2 Liveness: MCParReach FIXED in #456, dijkstra-mutex FIXED in #455
    # MultiPaxos_MC: FIXED in #472 (CHOOSE shadowing)
    # MC_HDiskSynod: FIXED (inline inner modules)
    # Paxos: FIXED in #460, MC_4: FIXED in #460, bosco: FIXED in #12
    # ACP_NB_TLC: Previously claimed FIXED in #530 but test FAILS as of M#100
    # VoucherIssue: FIXED in #531
    # product: FIXED in #529
    # AllocatorImplementation: FIXED in #54/6aa041b - 17701 states, no liveness error
    #
    # State counts: MCtcp #453, PaxosCommit #454, MCLamportMutex #463
    # MCInnerSerial: MOVED to TLA2_LIMITATIONS (performance, not bug)

    # === EWD998ChanID ===
    # #464: Fixed chained module reference parsing (A!B!C!D syntax)
    # #465: Fixed recursive function definition (nat2node[i \in S] == ...)
    # #466: Fixed SelectSeq lambda support (LAMBDA msg: msg.type = "pl")
    # #469: Fixed INSTANCE substitution scoping + liveness conversion for chained module refs
    #       (EWD998ChanID now matches TLC state count).
    # #470: Fixed LET in temporal properties (MCEWD687a's TreeWithRoot).

    # === BufferedRandomAccessFile ===
    # FIXED in #560: LET bindings with primed variables need substitution, not evaluation.
    # When a LET body references a primed variable (e.g., LET diskPosA == lo' IN ...),
    # we substitute the expression directly into the body instead of trying to evaluate it.
    # This allows the primed variable to be evaluated later when the next-state context
    # is available. Now matches TLC: 3316 states.

    # === Prisoner and PrisonerLightUnknown ===
    # FIXED: Both now pass. The fix for #14 (04156f5) also resolved these.
    # Conflict detection and preprocessing-disable-for-liveness fixed the false positives.

    # === ACTIVE BUGS ===
    # (none currently - MCCheckpointCoordination fixed by removing tla_library shadows)
}

# === HISTORICAL NOTES ===
# MCCheckpointCoordination: "Function application error" was caused by tla_library/FiniteSets.tla
# shadowing TLA2's built-in FiniteSets module. Fix: removed FiniteSets.tla and Bags.tla from
# tla_library (both shadow built-ins). FIXED in W#XXX (2026-01-12).

# Helper functions
def get_spec_by_name(name: str) -> Optional[SpecInfo]:
    """Get spec info by name."""
    for spec in ALL_SPECS:
        if spec.name == name:
            return spec
    return None

def get_specs_by_category(category: str) -> List[SpecInfo]:
    """Get all specs in a category."""
    return [s for s in ALL_SPECS if s.category == category]

def get_runnable_specs() -> List[SpecInfo]:
    """Get specs that can be run (not skipped)."""
    return [s for s in ALL_SPECS if s.name not in SKIP_SPECS]

def get_fast_specs() -> List[SpecInfo]:
    """Get specs that run quickly (<30s)."""
    return [s for s in get_runnable_specs() if s.name not in LARGE_SPECS]
