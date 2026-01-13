* [ ** GitHub ][1]

* [Home][2]
* Overview
* * [Key benefits][3]
  * [How does it work][4]
  * [Videos][5]
  * [Publications][6]
  * [FAQ][7]
* Get started with Coyote
* * [Installing Coyote][8]
  * [Using Coyote][9]
  * [Building from source][10]
  * [Telemetry][11]
* Tutorials
* * [Overview][12]
  * [Write your first concurrency unit test][13]
  * [Test concurrent CRUD operations][14]
  * Writing mocks
  * * [Mocking dependencies for testing][15]
    * [Simulating optimistic concurrency control][16]
    * [Testing an ASP.NET Core service][17]
  * [Test failover and liveness][18]
  * Actors and state machines
  * * [Hello world with actors][19]
    * [Raft actor service (on Azure)][20]
    * [Raft actor service (mocked)][21]
    * [Test failover and liveness][22]
* Concepts
* * [Program non-determinism][23]
  * [Concurrency unit testing][24]
  * [Binary rewriting for systematic testing][25]
  * [Program specifications][26]
  * Actors and state machines
  * * [Overview][27]
    * [State machines][28]
    * [Actor semantics][29]
    * [Actor termination][30]
    * [Using timers in actors][31]
    * [Event groups][32]
    * [Semantics of unhandled exceptions][33]
    * [Sharing objects][34]
    * [Logging][35]
    * [State machine demo][36]
    * [Why Coyote actors?][37]
* How-to guides
* * [Integrate with a unit testing framework][38]
  * [Find liveness bugs effectively][39]
  * [Track code and actor activity coverage][40]
* Samples
* * [Overview][41]
  * Task-based C# programs
  * * [Deadlock in bounded-buffer][42]
  * Actors and state machines
  * * [Bug in failure detector][43]
    * [Robot navigator failover][44]
* Case studies
* * [Azure Batch Service][45]
  * [Azure Blockchain Service][46]
  * [Azure Blockchain Workbench][47]
* API documentation
* * Microsoft.Coyote
  * * [Overview][48]
    * Microsoft.Coyote
    * * [Namespace Overview][49]
      * Configuration
      * * [Overview][50]
        * [Create][51]
        * [WithVerbosityEnabled][52]
        * [WithConsoleLoggingEnabled][53]
        * [WithTestingIterations][54]
        * [WithTestingTimeout][55]
        * [WithReproducibleTrace][56]
        * [WithRandomStrategy][57]
        * [WithProbabilisticStrategy][58]
        * [WithPrioritizationStrategy][59]
        * [WithDelayBoundingStrategy][60]
        * [WithQLearningStrategy][61]
        * [WithPartiallyControlledConcurrencyAllowed][62]
        * [WithPartiallyControlledDataNondeterminismAllowed][63]
        * [WithSystematicFuzzingEnabled][64]
        * [WithSystematicFuzzingFallbackEnabled][65]
        * [WithMaxFuzzingDelay][66]
        * [WithTraceAnalysisEnabled][67]
        * [WithCollectionAccessRaceCheckingEnabled][68]
        * [WithLockAccessRaceCheckingEnabled][69]
        * [WithAtomicOperationRaceCheckingEnabled][70]
        * [WithVolatileOperationRaceCheckingEnabled][71]
        * [WithMemoryAccessRaceCheckingEnabled][72]
        * [WithControlFlowRaceCheckingEnabled][73]
        * [WithExecutionTraceCycleReductionEnabled][74]
        * [WithPartialOrderSamplingEnabled][75]
        * [WithFailureOnMaxStepsBoundEnabled][76]
        * [WithNoBugTraceRepro][77]
        * [WithMaxSchedulingSteps][78]
        * [WithLivenessTemperatureThreshold][79]
        * [WithTimeoutDelay][80]
        * [WithDeadlockTimeout][81]
        * [WithPotentialDeadlocksReportedAsBugs][82]
        * [WithUncontrolledConcurrencyResolutionTimeout][83]
        * [WithRandomGeneratorSeed][84]
        * [WithUncontrolledInvocationStackTraceLoggingEnabled][85]
        * [WithActivityCoverageReported][86]
        * [WithScheduleCoverageReported][87]
        * [WithCoverageInfoSerialized][88]
        * [WithActorTraceVisualizationEnabled][89]
        * [WithXmlLogEnabled][90]
        * [WithTelemetryEnabled][91]
        * [Configuration][92]
        * [TestingIterations][93]
        * [RandomGeneratorSeed][94]
        * [MaxFuzzingDelay][95]
        * [MaxUnfairSchedulingSteps][96]
        * [MaxFairSchedulingSteps][97]
        * [TimeoutDelay][98]
        * [DeadlockTimeout][99]
        * [VerbosityLevel][100]
    * Microsoft.Coyote.Coverage
    * * [Namespace Overview][101]
      * CoverageGraph
      * * [Overview][102]
        * [GetNode][103]
        * [GetOrCreateNode][104]
        * [GetOrCreateLink][105]
        * [WriteDgml][106]
        * [LoadDgml][107]
        * [Merge][108]
        * [IsAvailable][109]
        * [ToString][110]
        * [CoverageGraph][111]
        * [Nodes][112]
        * [Links][113]
      * CoverageInfo
      * * [Overview][114]
        * [Load][115]
        * [Save][116]
        * [Merge][117]
        * [CoverageInfo][118]
        * [CoverageGraph][119]
        * [Monitors][120]
        * [MonitorsToStates][121]
        * [RegisteredMonitorEvents][122]
        * [MonitorEventInfo][123]
        * [SchedulingPointStackTraces][124]
        * [ExploredPaths][125]
        * [VisitedStates][126]
        * [Lock][127]
      * MonitorEventCoverage
      * * [Overview][128]
        * [GetEventsProcessed][129]
        * [GetEventsRaised][130]
        * [MonitorEventCoverage][131]
      * CoverageGraph.Object
      * * [Overview][132]
        * [AddAttribute][133]
        * [AddListAttribute][134]
        * [Object][135]
        * [Attributes][136]
        * [AttributeLists][137]
      * CoverageGraph.Node
      * * [Overview][138]
        * [AddDgmlProperties][139]
        * [Node][140]
        * [Id][141]
        * [Label][142]
        * [Category][143]
      * CoverageGraph.Link
      * * [Overview][144]
        * [AddDgmlProperties][145]
        * [Link][146]
        * [Label][147]
        * [Category][148]
        * [Source][149]
        * [Target][150]
        * [Index][151]
    * Microsoft.Coyote.Logging
    * * [Namespace Overview][152]
      * ConsoleLogger
      * * [Overview][153]
        * [Write][154]
        * [WriteLine][155]
        * [Dispose][156]
        * [ConsoleLogger][157]
      * ILogger
      * * [Overview][158]
        * [Write][159]
        * [WriteLine][160]
      * [LogSeverity][161]
      * MemoryLogger
      * * [Overview][162]
        * [Write][163]
        * [WriteLine][164]
        * [ToString][165]
        * [Dispose][166]
        * [MemoryLogger][167]
      * TextWriterLogger
      * * [Overview][168]
        * [Write][169]
        * [WriteLine][170]
        * [Dispose][171]
        * [TextWriterLogger][172]
      * [VerbosityLevel][173]
    * Microsoft.Coyote.Random
    * * [Namespace Overview][174]
      * Generator
      * * [Overview][175]
        * [Create][176]
        * [NextBoolean][177]
        * [NextInteger][178]
    * Microsoft.Coyote.Runtime
    * * [Namespace Overview][179]
      * [AssertionFailureException][180]
      * RuntimeException
      * * [Overview][181]
        * [RuntimeException][182]
      * IRuntimeExtension
      * * [Overview][183]
        * [RunTest][184]
        * [BuildCoverageInfo][185]
        * [GetCoverageInfo][186]
        * [GetCoverageGraph][187]
        * [WaitUntilQuiescenceAsync][188]
      * ICoyoteRuntime
      * * [Overview][189]
        * [RegisterMonitor][190]
        * [Monitor][191]
        * [RandomBoolean][192]
        * [RandomInteger][193]
        * [Assert][194]
        * [RegisterLog][195]
        * [RemoveLog][196]
        * [Stop][197]
        * [Logger][198]
        * [OnFailure][199]
      * IRuntimeLog
      * * [Overview][200]
        * [OnCreateMonitor][201]
        * [OnMonitorExecuteAction][202]
        * [OnMonitorProcessEvent][203]
        * [OnMonitorRaiseEvent][204]
        * [OnMonitorStateTransition][205]
        * [OnMonitorError][206]
        * [OnRandom][207]
        * [OnAssertionFailure][208]
        * [OnCompleted][209]
      * RuntimeLogTextFormatter
      * * [Overview][210]
        * [OnCreateMonitor][211]
        * [OnMonitorExecuteAction][212]
        * [OnMonitorProcessEvent][213]
        * [OnMonitorRaiseEvent][214]
        * [OnMonitorStateTransition][215]
        * [OnMonitorError][216]
        * [OnRandom][217]
        * [OnAssertionFailure][218]
        * [OnCompleted][219]
        * [RuntimeLogTextFormatter][220]
        * [Logger][221]
      * [OnFailureHandler][222]
      * IOperationBuilder
      * * [Overview][223]
        * [Name][224]
        * [GroupId][225]
        * [HashedStateCallback][226]
      * Operation
      * * [Overview][227]
        * [GetNextId][228]
        * [CreateNext][229]
        * [CreateFrom][230]
        * [Start][231]
        * [PauseUntil][232]
        * [PauseUntilCompleted][233]
        * [PauseUntilAsync][234]
        * [PauseUntilCompletedAsync][235]
        * [ScheduleNext][236]
        * [OnStarted][237]
        * [OnCompleted][238]
        * [TryReset][239]
        * [RegisterCallSite][240]
      * RuntimeProvider
      * * [Overview][241]
        * [Current][242]
      * SchedulingPoint
      * * [Overview][243]
        * [Interleave][244]
        * [Yield][245]
        * [Read][246]
        * [Write][247]
        * [Suppress][248]
        * [Resume][249]
        * [SetCheckpoint][250]
      * [TaskServices][251]
    * Microsoft.Coyote.Specifications
    * * [Namespace Overview][252]
      * Monitor
      * * [Overview][253]
        * [RaiseEvent][254]
        * [RaiseGotoStateEvent][255]
        * [Assert][256]
        * [ToString][257]
        * [Monitor][258]
        * [Logger][259]
        * [CurrentState][260]
        * [HashedState][261]
      * Specification
      * * [Overview][262]
        * [Assert][263]
        * [IsEventuallyCompletedSuccessfully][264]
        * [RegisterMonitor][265]
        * [Monitor][266]
        * [RegisterStateHashingFunction][267]
      * Monitor.State
      * * [Overview][268]
        * [State][269]
      * Monitor.StateGroup
      * * [Overview][270]
        * [StateGroup][271]
      * Monitor.Event
      * * [Overview][272]
        * [Event][273]
      * Monitor.WildCardEvent
      * * [Overview][274]
        * [WildCardEvent][275]
      * Monitor.State.StartAttribute
      * * [Overview][276]
        * [StartAttribute][277]
      * Monitor.State.OnEntryAttribute
      * * [Overview][278]
        * [OnEntryAttribute][279]
      * Monitor.State.OnExitAttribute
      * * [Overview][280]
        * [OnExitAttribute][281]
      * Monitor.State.OnEventGotoStateAttribute
      * * [Overview][282]
        * [OnEventGotoStateAttribute][283]
      * Monitor.State.OnEventDoActionAttribute
      * * [Overview][284]
        * [OnEventDoActionAttribute][285]
      * Monitor.State.IgnoreEventsAttribute
      * * [Overview][286]
        * [IgnoreEventsAttribute][287]
      * Monitor.State.ColdAttribute
      * * [Overview][288]
        * [ColdAttribute][289]
      * Monitor.State.HotAttribute
      * * [Overview][290]
        * [HotAttribute][291]
  * Microsoft.Coyote.Actors
  * * [Overview][292]
    * Microsoft.Coyote.Actors
    * * [Namespace Overview][293]
      * [DequeueStatus][294]
      * AwaitableEventGroup
      * * [Overview][295]
        * [SetResult][296]
        * [TrySetResult][297]
        * [SetCancelled][298]
        * [TrySetCanceled][299]
        * [SetException][300]
        * [TrySetException][301]
        * [GetAwaiter][302]
        * [AwaitableEventGroup][303]
        * [Task][304]
        * [IsCompleted][305]
        * [IsCanceled][306]
        * [IsFaulted][307]
      * DefaultEvent
      * * [Overview][308]
        * [Instance][309]
      * Event
      * * [Overview][310]
        * [Event][311]
      * EventGroup
      * * [Overview][312]
        * [EventGroup][313]
        * [Id][314]
        * [Name][315]
      * HaltEvent
      * * [Overview][316]
        * [Instance][317]
      * WildCardEvent
      * * [Overview][318]
        * [WildCardEvent][319]
      * [OnExceptionOutcome][320]
      * UnhandledEventException
      * * [Overview][321]
        * [UnhandledEvent][322]
        * [CurrentStateName][323]
      * [OnActorHaltedHandler][324]
      * [OnEventDroppedHandler][325]
      * ActorRuntimeLogTextFormatter
      * * [Overview][326]
        * [OnCreateActor][327]
        * [OnCreateStateMachine][328]
        * [OnCreateTimer][329]
        * [OnDefaultEventHandler][330]
        * [OnEventHandlerTerminated][331]
        * [OnDequeueEvent][332]
        * [OnEnqueueEvent][333]
        * [OnExceptionHandled][334]
        * [OnExceptionThrown][335]
        * [OnExecuteAction][336]
        * [OnGotoState][337]
        * [OnHalt][338]
        * [OnPopState][339]
        * [OnPopStateUnhandledEvent][340]
        * [OnPushState][341]
        * [OnRaiseEvent][342]
        * [OnHandleRaisedEvent][343]
        * [OnReceiveEvent][344]
        * [OnSendEvent][345]
        * [OnStateTransition][346]
        * [OnStopTimer][347]
        * [OnWaitEvent][348]
        * [ActorRuntimeLogTextFormatter][349]
      * IActorRuntimeLog
      * * [Overview][350]
        * [OnCreateActor][351]
        * [OnCreateStateMachine][352]
        * [OnExecuteAction][353]
        * [OnSendEvent][354]
        * [OnRaiseEvent][355]
        * [OnHandleRaisedEvent][356]
        * [OnEnqueueEvent][357]
        * [OnDequeueEvent][358]
        * [OnReceiveEvent][359]
        * [OnWaitEvent][360]
        * [OnStateTransition][361]
        * [OnGotoState][362]
        * [OnPushState][363]
        * [OnPopState][364]
        * [OnDefaultEventHandler][365]
        * [OnEventHandlerTerminated][366]
        * [OnHalt][367]
        * [OnPopStateUnhandledEvent][368]
        * [OnExceptionThrown][369]
        * [OnExceptionHandled][370]
        * [OnCreateTimer][371]
        * [OnStopTimer][372]
      * Actor
      * * [Overview][373]
        * [CreateActor][374]
        * [SendEvent][375]
        * [ReceiveEventAsync][376]
        * [StartTimer][377]
        * [StartPeriodicTimer][378]
        * [StopTimer][379]
        * [RandomBoolean][380]
        * [RandomInteger][381]
        * [Monitor][382]
        * [Assert][383]
        * [RaiseHaltEvent][384]
        * [OnInitializeAsync][385]
        * [OnEventDequeuedAsync][386]
        * [OnEventIgnored][387]
        * [OnEventDeferred][388]
        * [OnEventHandledAsync][389]
        * [OnEventUnhandledAsync][390]
        * [OnExceptionHandledAsync][391]
        * [OnHaltAsync][392]
        * [OnException][393]
        * [Equals][394]
        * [GetHashCode][395]
        * [ToString][396]
        * [Actor][397]
        * [Id][398]
        * [CurrentEventGroup][399]
        * [Logger][400]
        * [HashedState][401]
      * [ActorExecutionStatus][402]
      * ActorId
      * * [Overview][403]
        * [Equals][404]
        * [GetHashCode][405]
        * [ToString][406]
        * [CompareTo][407]
        * [Runtime][408]
        * [IsNameUsedForHashing][409]
        * [Value][410]
        * [NameValue][411]
        * [Type][412]
        * [Name][413]
      * IActorRuntime
      * * [Overview][414]
        * [CreateActorId][415]
        * [CreateActorIdFromName][416]
        * [CreateActor][417]
        * [SendEvent][418]
        * [GetCurrentEventGroup][419]
        * [GetActorExecutionStatus][420]
        * [GetCurrentActorIds][421]
        * [GetCurrentActorTypes][422]
        * [GetCurrentActorCount][423]
        * [OnActorHalted][424]
        * [OnEventDropped][425]
      * RuntimeFactory
      * * [Overview][426]
        * [Create][427]
      * SendOptions
      * * [Overview][428]
        * [ToString][429]
        * [SendOptions][430]
        * [Default][431]
        * [MustHandle][432]
        * [Assert][433]
        * [HashedState][434]
      * StateMachine
      * * [Overview][435]
        * [RaiseEvent][436]
        * [RaiseGotoStateEvent][437]
        * [RaisePushStateEvent][438]
        * [RaisePopStateEvent][439]
        * [RaiseHaltEvent][440]
        * [OnEventHandledAsync][441]
        * [StateMachine][442]
        * [CurrentState][443]
      * Actor.OnEventDoActionAttribute
      * * [Overview][444]
        * [OnEventDoActionAttribute][445]
      * StateMachine.State
      * * [Overview][446]
        * [State][447]
      * StateMachine.StateGroup
      * * [Overview][448]
        * [StateGroup][449]
      * StateMachine.State.StartAttribute
      * * [Overview][450]
        * [StartAttribute][451]
      * StateMachine.State.OnEntryAttribute
      * * [Overview][452]
        * [OnEntryAttribute][453]
      * StateMachine.State.OnExitAttribute
      * * [Overview][454]
        * [OnExitAttribute][455]
      * StateMachine.State.OnEventGotoStateAttribute
      * * [Overview][456]
        * [OnEventGotoStateAttribute][457]
      * StateMachine.State.OnEventPushStateAttribute
      * * [Overview][458]
        * [OnEventPushStateAttribute][459]
      * StateMachine.State.OnEventDoActionAttribute
      * * [Overview][460]
        * [OnEventDoActionAttribute][461]
      * StateMachine.State.DeferEventsAttribute
      * * [Overview][462]
        * [DeferEventsAttribute][463]
      * StateMachine.State.IgnoreEventsAttribute
      * * [Overview][464]
        * [IgnoreEventsAttribute][465]
    * Microsoft.Coyote.Actors.Coverage
    * * [Namespace Overview][466]
      * ActorCoverageInfo
      * * [Overview][467]
        * [IsMachineDeclared][468]
        * [DeclareMachineState][469]
        * [DeclareMachineStateEventPair][470]
        * [Merge][471]
        * [ActorCoverageInfo][472]
        * [Machines][473]
        * [MachinesToStates][474]
        * [RegisteredActorEvents][475]
        * [ActorEventInfo][476]
      * ActorEventCoverage
      * * [Overview][477]
        * [GetEventsReceived][478]
        * [GetEventsSent][479]
        * [ActorEventCoverage][480]
    * Microsoft.Coyote.Actors.SharedObjects
    * * [Namespace Overview][481]
      * SharedCounter
      * * [Overview][482]
        * [Create][483]
        * [Increment][484]
        * [Decrement][485]
        * [GetValue][486]
        * [Add][487]
        * [Exchange][488]
        * [CompareExchange][489]
      * SharedDictionary
      * * [Overview][490]
        * [Create][491]
      * SharedDictionary
      * * [Overview][492]
        * [TryAdd][493]
        * [TryUpdate][494]
        * [TryGetValue][495]
        * [TryRemove][496]
        * [Item][497]
        * [Count][498]
      * SharedRegister
      * * [Overview][499]
        * [Create][500]
      * SharedRegister
      * * [Overview][501]
        * [Update][502]
        * [GetValue][503]
        * [SetValue][504]
    * Microsoft.Coyote.Actors.Timers
    * * [Namespace Overview][505]
      * TimerElapsedEvent
      * * [Overview][506]
        * [TimerElapsedEvent][507]
        * [Info][508]
      * TimerInfo
      * * [Overview][509]
        * [Equals][510]
        * [GetHashCode][511]
        * [ToString][512]
        * [OwnerId][513]
        * [DueTime][514]
        * [Period][515]
        * [CustomEvent][516]
    * Microsoft.Coyote.Actors.UnitTesting
    * * [Namespace Overview][517]
      * ActorTestKit
      * * [Overview][518]
        * [StartActorAsync][519]
        * [SendEventAsync][520]
        * [Invoke][521]
        * [InvokeAsync][522]
        * [Assert][523]
        * [AssertStateTransition][524]
        * [AssertIsWaitingToReceiveEvent][525]
        * [AssertInboxSize][526]
        * [ActorTestKit][527]
        * [Logger][528]
        * [ActorInstance][529]
  * Microsoft.Coyote.Test
  * * [Overview][530]
    * Microsoft.Coyote.Rewriting
    * * [Namespace Overview][531]
      * RewritingSignatureAttribute
      * * [Overview][532]
        * [RewritingSignatureAttribute][533]
        * [Version][534]
        * [Signature][535]
      * SkipRewritingAttribute
      * * [Overview][536]
        * [SkipRewritingAttribute][537]
        * [Reason][538]
      * RewritingEngine
      * * [Overview][539]
        * [IsAssemblyRewritten][540]
    * Microsoft.Coyote.SystematicTesting
    * * [Namespace Overview][541]
      * TestReport
      * * [Overview][542]
        * [Merge][543]
        * [GetText][544]
        * [Clone][545]
        * [TestReport][546]
        * [Configuration][547]
        * [CoverageInfo][548]
        * [NumOfExploredFairPaths][549]
        * [NumOfExploredUnfairPaths][550]
        * [NumOfFoundBugs][551]
        * [BugReports][552]
        * [UncontrolledInvocations][553]
        * [MinControlledOperations][554]
        * [MaxControlledOperations][555]
        * [TotalControlledOperations][556]
        * [MinConcurrencyDegree][557]
        * [MaxConcurrencyDegree][558]
        * [TotalConcurrencyDegree][559]
        * [MinOperationGroupingDegree][560]
        * [MaxOperationGroupingDegree][561]
        * [TotalOperationGroupingDegree][562]
        * [MinExploredFairSteps][563]
        * [MaxExploredFairSteps][564]
        * [TotalExploredFairSteps][565]
        * [MinExploredUnfairSteps][566]
        * [MaxExploredUnfairSteps][567]
        * [TotalExploredUnfairSteps][568]
        * [MaxFairStepsHitInFairTests][569]
        * [MaxUnfairStepsHitInFairTests][570]
        * [MaxUnfairStepsHitInUnfairTests][571]
        * [InternalErrors][572]
      * TestAttribute
      * * [Overview][573]
        * [TestAttribute][574]
      * TestInitAttribute
      * * [Overview][575]
        * [TestInitAttribute][576]
      * TestDisposeAttribute
      * * [Overview][577]
        * [TestDisposeAttribute][578]
      * TestIterationDisposeAttribute
      * * [Overview][579]
        * [TestIterationDisposeAttribute][580]
      * TestingEngine
      * * [Overview][581]
        * [Create][582]
        * [Run][583]
        * [Stop][584]
        * [GetReport][585]
        * [ThrowIfBugFound][586]
        * [TryEmitReports][587]
        * [TryEmitCoverageReports][588]
        * [RegisterStartIterationCallBack][589]
        * [RegisterEndIterationCallBack][590]
        * [InvokeStartIterationCallBacks][591]
        * [InvokeEndIterationCallBacks][592]
        * [IsTestRewritten][593]
        * [SetLogger][594]
        * [Dispose][595]
        * [TestReport][596]
        * [ReadableTrace][597]
        * [ReproducibleTrace][598]
    * Microsoft.Coyote.SystematicTesting.Frameworks.XUnit
    * * [Namespace Overview][599]
      * TestOutputLogger
      * * [Overview][600]
        * [Write][601]
        * [WriteLine][602]
        * [Dispose][603]
        * [TestOutputLogger][604]
    * Microsoft.Coyote.Web
    * * [Namespace Overview][605]
      * RequestControllerMiddlewareExtensions
      * * [Overview][606]
        * [UseRequestController][607]

[1]: https://github.com/microsoft/coyote/
[2]: 
[3]: overview/benefits/
[4]: overview/how/
[5]: overview/videos/
[6]: overview/publications/
[7]: overview/faq/
[8]: get-started/install/
[9]: get-started/using-coyote/
[10]: get-started/build-source/
[11]: get-started/telemetry/
[12]: tutorials/overview/
[13]: tutorials/first-concurrency-unit-test/
[14]: tutorials/test-concurrent-operations/
[15]: tutorials/mocks/mock-dependencies/
[16]: tutorials/mocks/optimistic-concurrency-control/
[17]: tutorials/testing-aspnet-service/
[18]: tutorials/test-failover/
[19]: tutorials/actors/hello-world/
[20]: tutorials/actors/raft-azure/
[21]: tutorials/actors/raft-mocking/
[22]: tutorials/actors/test-failover/
[23]: concepts/non-determinism/
[24]: concepts/concurrency-unit-testing/
[25]: concepts/binary-rewriting/
[26]: concepts/specifications/
[27]: concepts/actors/overview/
[28]: concepts/actors/state-machines/
[29]: concepts/actors/actor-semantics/
[30]: concepts/actors/termination/
[31]: concepts/actors/timers/
[32]: concepts/actors/event-groups/
[33]: concepts/actors/uncaught-exceptions/
[34]: concepts/actors/sharing-objects/
[35]: concepts/actors/logging/
[36]: concepts/actors/state-machine-demo/
[37]: concepts/actors/why-actors/
[38]: how-to/unit-testing/
[39]: how-to/liveness-checking/
[40]: how-to/coverage/
[41]: samples/overview/
[42]: samples/tasks/bounded-buffer/
[43]: samples/actors/failure-detector/
[44]: samples/actors/failover-robot-navigator/
[45]: case-studies/azure-batch-service/
[46]: case-studies/azure-blockchain-service/
[47]: case-studies/azure-blockchain-workbench/
[48]: ref/Microsoft.Coyote/
[49]: ref/Microsoft.CoyoteNamespace/
[50]: ref/Microsoft.Coyote/Configuration/
[51]: ref/Microsoft.Coyote/Configuration/Create/
[52]: ref/Microsoft.Coyote/Configuration/WithVerbosityEnabled/
[53]: ref/Microsoft.Coyote/Configuration/WithConsoleLoggingEnabled/
[54]: ref/Microsoft.Coyote/Configuration/WithTestingIterations/
[55]: ref/Microsoft.Coyote/Configuration/WithTestingTimeout/
[56]: ref/Microsoft.Coyote/Configuration/WithReproducibleTrace/
[57]: ref/Microsoft.Coyote/Configuration/WithRandomStrategy/
[58]: ref/Microsoft.Coyote/Configuration/WithProbabilisticStrategy/
[59]: ref/Microsoft.Coyote/Configuration/WithPrioritizationStrategy/
[60]: ref/Microsoft.Coyote/Configuration/WithDelayBoundingStrategy/
[61]: ref/Microsoft.Coyote/Configuration/WithQLearningStrategy/
[62]: ref/Microsoft.Coyote/Configuration/WithPartiallyControlledConcurrencyAllowed/
[63]: ref/Microsoft.Coyote/Configuration/WithPartiallyControlledDataNondeterminismAllowed/
[64]: ref/Microsoft.Coyote/Configuration/WithSystematicFuzzingEnabled/
[65]: ref/Microsoft.Coyote/Configuration/WithSystematicFuzzingFallbackEnabled/
[66]: ref/Microsoft.Coyote/Configuration/WithMaxFuzzingDelay/
[67]: ref/Microsoft.Coyote/Configuration/WithTraceAnalysisEnabled/
[68]: ref/Microsoft.Coyote/Configuration/WithCollectionAccessRaceCheckingEnabled/
[69]: ref/Microsoft.Coyote/Configuration/WithLockAccessRaceCheckingEnabled/
[70]: ref/Microsoft.Coyote/Configuration/WithAtomicOperationRaceCheckingEnabled/
[71]: ref/Microsoft.Coyote/Configuration/WithVolatileOperationRaceCheckingEnabled/
[72]: ref/Microsoft.Coyote/Configuration/WithMemoryAccessRaceCheckingEnabled/
[73]: ref/Microsoft.Coyote/Configuration/WithControlFlowRaceCheckingEnabled/
[74]: ref/Microsoft.Coyote/Configuration/WithExecutionTraceCycleReductionEnabled/
[75]: ref/Microsoft.Coyote/Configuration/WithPartialOrderSamplingEnabled/
[76]: ref/Microsoft.Coyote/Configuration/WithFailureOnMaxStepsBoundEnabled/
[77]: ref/Microsoft.Coyote/Configuration/WithNoBugTraceRepro/
[78]: ref/Microsoft.Coyote/Configuration/WithMaxSchedulingSteps/
[79]: ref/Microsoft.Coyote/Configuration/WithLivenessTemperatureThreshold/
[80]: ref/Microsoft.Coyote/Configuration/WithTimeoutDelay/
[81]: ref/Microsoft.Coyote/Configuration/WithDeadlockTimeout/
[82]: ref/Microsoft.Coyote/Configuration/WithPotentialDeadlocksReportedAsBugs/
[83]: ref/Microsoft.Coyote/Configuration/WithUncontrolledConcurrencyResolutionTimeout/
[84]: ref/Microsoft.Coyote/Configuration/WithRandomGeneratorSeed/
[85]: ref/Microsoft.Coyote/Configuration/WithUncontrolledInvocationStackTraceLoggingEnabled/
[86]: ref/Microsoft.Coyote/Configuration/WithActivityCoverageReported/
[87]: ref/Microsoft.Coyote/Configuration/WithScheduleCoverageReported/
[88]: ref/Microsoft.Coyote/Configuration/WithCoverageInfoSerialized/
[89]: ref/Microsoft.Coyote/Configuration/WithActorTraceVisualizationEnabled/
[90]: ref/Microsoft.Coyote/Configuration/WithXmlLogEnabled/
[91]: ref/Microsoft.Coyote/Configuration/WithTelemetryEnabled/
[92]: ref/Microsoft.Coyote/Configuration/Configuration/
[93]: ref/Microsoft.Coyote/Configuration/TestingIterations/
[94]: ref/Microsoft.Coyote/Configuration/RandomGeneratorSeed/
[95]: ref/Microsoft.Coyote/Configuration/MaxFuzzingDelay/
[96]: ref/Microsoft.Coyote/Configuration/MaxUnfairSchedulingSteps/
[97]: ref/Microsoft.Coyote/Configuration/MaxFairSchedulingSteps/
[98]: ref/Microsoft.Coyote/Configuration/TimeoutDelay/
[99]: ref/Microsoft.Coyote/Configuration/DeadlockTimeout/
[100]: ref/Microsoft.Coyote/Configuration/VerbosityLevel/
[101]: ref/Microsoft.Coyote.CoverageNamespace/
[102]: ref/Microsoft.Coyote.Coverage/CoverageGraph.md
[103]: ref/Microsoft.Coyote.Coverage/CoverageGraph/GetNode.md
[104]: ref/Microsoft.Coyote.Coverage/CoverageGraph/GetOrCreateNode.md
[105]: ref/Microsoft.Coyote.Coverage/CoverageGraph/GetOrCreateLink.md
[106]: ref/Microsoft.Coyote.Coverage/CoverageGraph/WriteDgml.md
[107]: ref/Microsoft.Coyote.Coverage/CoverageGraph/LoadDgml.md
[108]: ref/Microsoft.Coyote.Coverage/CoverageGraph/Merge.md
[109]: ref/Microsoft.Coyote.Coverage/CoverageGraph/IsAvailable.md
[110]: ref/Microsoft.Coyote.Coverage/CoverageGraph/ToString.md
[111]: ref/Microsoft.Coyote.Coverage/CoverageGraph/CoverageGraph.md
[112]: ref/Microsoft.Coyote.Coverage/CoverageGraph/Nodes.md
[113]: ref/Microsoft.Coyote.Coverage/CoverageGraph/Links.md
[114]: ref/Microsoft.Coyote.Coverage/CoverageInfo.md
[115]: ref/Microsoft.Coyote.Coverage/CoverageInfo/Load.md
[116]: ref/Microsoft.Coyote.Coverage/CoverageInfo/Save.md
[117]: ref/Microsoft.Coyote.Coverage/CoverageInfo/Merge.md
[118]: ref/Microsoft.Coyote.Coverage/CoverageInfo/CoverageInfo.md
[119]: ref/Microsoft.Coyote.Coverage/CoverageInfo/CoverageGraph.md
[120]: ref/Microsoft.Coyote.Coverage/CoverageInfo/Monitors.md
[121]: ref/Microsoft.Coyote.Coverage/CoverageInfo/MonitorsToStates.md
[122]: ref/Microsoft.Coyote.Coverage/CoverageInfo/RegisteredMonitorEvents.md
[123]: ref/Microsoft.Coyote.Coverage/CoverageInfo/MonitorEventInfo.md
[124]: ref/Microsoft.Coyote.Coverage/CoverageInfo/SchedulingPointStackTraces.md
[125]: ref/Microsoft.Coyote.Coverage/CoverageInfo/ExploredPaths.md
[126]: ref/Microsoft.Coyote.Coverage/CoverageInfo/VisitedStates.md
[127]: ref/Microsoft.Coyote.Coverage/CoverageInfo/Lock.md
[128]: ref/Microsoft.Coyote.Coverage/MonitorEventCoverage.md
[129]: ref/Microsoft.Coyote.Coverage/MonitorEventCoverage/GetEventsProcessed.md
[130]: ref/Microsoft.Coyote.Coverage/MonitorEventCoverage/GetEventsRaised.md
[131]: ref/Microsoft.Coyote.Coverage/MonitorEventCoverage/MonitorEventCoverage.md
[132]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Object.md
[133]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Object/AddAttribute.md
[134]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Object/AddListAttribute.md
[135]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Object/Object.md
[136]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Object/Attributes.md
[137]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Object/AttributeLists.md
[138]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Node.md
[139]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Node/AddDgmlProperties.md
[140]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Node/Node.md
[141]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Node/Id.md
[142]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Node/Label.md
[143]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Node/Category.md
[144]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Link.md
[145]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Link/AddDgmlProperties.md
[146]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Link/Link.md
[147]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Link/Label.md
[148]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Link/Category.md
[149]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Link/Source.md
[150]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Link/Target.md
[151]: ref/Microsoft.Coyote.Coverage/CoverageGraph.Link/Index.md
[152]: ref/Microsoft.Coyote.LoggingNamespace/
[153]: ref/Microsoft.Coyote.Logging/ConsoleLogger/
[154]: ref/Microsoft.Coyote.Logging/ConsoleLogger/Write/
[155]: ref/Microsoft.Coyote.Logging/ConsoleLogger/WriteLine/
[156]: ref/Microsoft.Coyote.Logging/ConsoleLogger/Dispose/
[157]: ref/Microsoft.Coyote.Logging/ConsoleLogger/ConsoleLogger/
[158]: ref/Microsoft.Coyote.Logging/ILogger/
[159]: ref/Microsoft.Coyote.Logging/ILogger/Write/
[160]: ref/Microsoft.Coyote.Logging/ILogger/WriteLine/
[161]: ref/Microsoft.Coyote.Logging/LogSeverity/
[162]: ref/Microsoft.Coyote.Logging/MemoryLogger/
[163]: ref/Microsoft.Coyote.Logging/MemoryLogger/Write/
[164]: ref/Microsoft.Coyote.Logging/MemoryLogger/WriteLine/
[165]: ref/Microsoft.Coyote.Logging/MemoryLogger/ToString/
[166]: ref/Microsoft.Coyote.Logging/MemoryLogger/Dispose/
[167]: ref/Microsoft.Coyote.Logging/MemoryLogger/MemoryLogger/
[168]: ref/Microsoft.Coyote.Logging/TextWriterLogger/
[169]: ref/Microsoft.Coyote.Logging/TextWriterLogger/Write/
[170]: ref/Microsoft.Coyote.Logging/TextWriterLogger/WriteLine/
[171]: ref/Microsoft.Coyote.Logging/TextWriterLogger/Dispose/
[172]: ref/Microsoft.Coyote.Logging/TextWriterLogger/TextWriterLogger/
[173]: ref/Microsoft.Coyote.Logging/VerbosityLevel/
[174]: ref/Microsoft.Coyote.RandomNamespace/
[175]: ref/Microsoft.Coyote.Random/Generator/
[176]: ref/Microsoft.Coyote.Random/Generator/Create/
[177]: ref/Microsoft.Coyote.Random/Generator/NextBoolean/
[178]: ref/Microsoft.Coyote.Random/Generator/NextInteger/
[179]: ref/Microsoft.Coyote.RuntimeNamespace/
[180]: ref/Microsoft.Coyote.Runtime/AssertionFailureException/
[181]: ref/Microsoft.Coyote.Runtime/RuntimeException/
[182]: ref/Microsoft.Coyote.Runtime/RuntimeException/RuntimeException/
[183]: ref/Microsoft.Coyote.Runtime/IRuntimeExtension/
[184]: ref/Microsoft.Coyote.Runtime/IRuntimeExtension/RunTest/
[185]: ref/Microsoft.Coyote.Runtime/IRuntimeExtension/BuildCoverageInfo/
[186]: ref/Microsoft.Coyote.Runtime/IRuntimeExtension/GetCoverageInfo/
[187]: ref/Microsoft.Coyote.Runtime/IRuntimeExtension/GetCoverageGraph/
[188]: ref/Microsoft.Coyote.Runtime/IRuntimeExtension/WaitUntilQuiescenceAsync/
[189]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/
[190]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/RegisterMonitor/
[191]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/Monitor/
[192]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/RandomBoolean/
[193]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/RandomInteger/
[194]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/Assert/
[195]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/RegisterLog/
[196]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/RemoveLog/
[197]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/Stop/
[198]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/Logger/
[199]: ref/Microsoft.Coyote.Runtime/ICoyoteRuntime/OnFailure/
[200]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/
[201]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/OnCreateMonitor/
[202]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/OnMonitorExecuteAction/
[203]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/OnMonitorProcessEvent/
[204]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/OnMonitorRaiseEvent/
[205]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/OnMonitorStateTransition/
[206]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/OnMonitorError/
[207]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/OnRandom/
[208]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/OnAssertionFailure/
[209]: ref/Microsoft.Coyote.Runtime/IRuntimeLog/OnCompleted/
[210]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/
[211]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/OnCreateMonitor/
[212]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/OnMonitorExecuteAction/
[213]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/OnMonitorProcessEvent/
[214]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/OnMonitorRaiseEvent/
[215]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/OnMonitorStateTransition/
[216]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/OnMonitorError/
[217]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/OnRandom/
[218]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/OnAssertionFailure/
[219]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/OnCompleted/
[220]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/RuntimeLogTextFormatter/
[221]: ref/Microsoft.Coyote.Runtime/RuntimeLogTextFormatter/Logger/
[222]: ref/Microsoft.Coyote.Runtime/OnFailureHandler/
[223]: ref/Microsoft.Coyote.Runtime/IOperationBuilder/
[224]: ref/Microsoft.Coyote.Runtime/IOperationBuilder/Name/
[225]: ref/Microsoft.Coyote.Runtime/IOperationBuilder/GroupId/
[226]: ref/Microsoft.Coyote.Runtime/IOperationBuilder/HashedStateCallback/
[227]: ref/Microsoft.Coyote.Runtime/Operation/
[228]: ref/Microsoft.Coyote.Runtime/Operation/GetNextId/
[229]: ref/Microsoft.Coyote.Runtime/Operation/CreateNext/
[230]: ref/Microsoft.Coyote.Runtime/Operation/CreateFrom/
[231]: ref/Microsoft.Coyote.Runtime/Operation/Start/
[232]: ref/Microsoft.Coyote.Runtime/Operation/PauseUntil/
[233]: ref/Microsoft.Coyote.Runtime/Operation/PauseUntilCompleted/
[234]: ref/Microsoft.Coyote.Runtime/Operation/PauseUntilAsync/
[235]: ref/Microsoft.Coyote.Runtime/Operation/PauseUntilCompletedAsync/
[236]: ref/Microsoft.Coyote.Runtime/Operation/ScheduleNext/
[237]: ref/Microsoft.Coyote.Runtime/Operation/OnStarted/
[238]: ref/Microsoft.Coyote.Runtime/Operation/OnCompleted/
[239]: ref/Microsoft.Coyote.Runtime/Operation/TryReset/
[240]: ref/Microsoft.Coyote.Runtime/Operation/RegisterCallSite/
[241]: ref/Microsoft.Coyote.Runtime/RuntimeProvider/
[242]: ref/Microsoft.Coyote.Runtime/RuntimeProvider/Current/
[243]: ref/Microsoft.Coyote.Runtime/SchedulingPoint/
[244]: ref/Microsoft.Coyote.Runtime/SchedulingPoint/Interleave/
[245]: ref/Microsoft.Coyote.Runtime/SchedulingPoint/Yield/
[246]: ref/Microsoft.Coyote.Runtime/SchedulingPoint/Read/
[247]: ref/Microsoft.Coyote.Runtime/SchedulingPoint/Write/
[248]: ref/Microsoft.Coyote.Runtime/SchedulingPoint/Suppress/
[249]: ref/Microsoft.Coyote.Runtime/SchedulingPoint/Resume/
[250]: ref/Microsoft.Coyote.Runtime/SchedulingPoint/SetCheckpoint/
[251]: ref/Microsoft.Coyote.Runtime/TaskServices/
[252]: ref/Microsoft.Coyote.SpecificationsNamespace/
[253]: ref/Microsoft.Coyote.Specifications/Monitor/
[254]: ref/Microsoft.Coyote.Specifications/Monitor/RaiseEvent/
[255]: ref/Microsoft.Coyote.Specifications/Monitor/RaiseGotoStateEvent/
[256]: ref/Microsoft.Coyote.Specifications/Monitor/Assert/
[257]: ref/Microsoft.Coyote.Specifications/Monitor/ToString/
[258]: ref/Microsoft.Coyote.Specifications/Monitor/Monitor/
[259]: ref/Microsoft.Coyote.Specifications/Monitor/Logger/
[260]: ref/Microsoft.Coyote.Specifications/Monitor/CurrentState/
[261]: ref/Microsoft.Coyote.Specifications/Monitor/HashedState/
[262]: ref/Microsoft.Coyote.Specifications/Specification/
[263]: ref/Microsoft.Coyote.Specifications/Specification/Assert/
[264]: ref/Microsoft.Coyote.Specifications/Specification/IsEventuallyCompletedSuccessfully/
[265]: ref/Microsoft.Coyote.Specifications/Specification/RegisterMonitor/
[266]: ref/Microsoft.Coyote.Specifications/Specification/Monitor/
[267]: ref/Microsoft.Coyote.Specifications/Specification/RegisterStateHashingFunction/
[268]: ref/Microsoft.Coyote.Specifications/Monitor.State/
[269]: ref/Microsoft.Coyote.Specifications/Monitor.State/State/
[270]: ref/Microsoft.Coyote.Specifications/Monitor.StateGroup/
[271]: ref/Microsoft.Coyote.Specifications/Monitor.StateGroup/StateGroup/
[272]: ref/Microsoft.Coyote.Specifications/Monitor.Event/
[273]: ref/Microsoft.Coyote.Specifications/Monitor.Event/Event/
[274]: ref/Microsoft.Coyote.Specifications/Monitor.WildCardEvent/
[275]: ref/Microsoft.Coyote.Specifications/Monitor.WildCardEvent/WildCardEvent/
[276]: ref/Microsoft.Coyote.Specifications/Monitor.State.StartAttribute/
[277]: ref/Microsoft.Coyote.Specifications/Monitor.State.StartAttribute/StartAttribute/
[278]: ref/Microsoft.Coyote.Specifications/Monitor.State.OnEntryAttribute/
[279]: ref/Microsoft.Coyote.Specifications/Monitor.State.OnEntryAttribute/OnEntryAttribute/
[280]: ref/Microsoft.Coyote.Specifications/Monitor.State.OnExitAttribute/
[281]: ref/Microsoft.Coyote.Specifications/Monitor.State.OnExitAttribute/OnExitAttribute/
[282]: ref/Microsoft.Coyote.Specifications/Monitor.State.OnEventGotoStateAttribute/
[283]: ref/Microsoft.Coyote.Specifications/Monitor.State.OnEventGotoStateAttribute/OnEventGotoStateA
ttribute/
[284]: ref/Microsoft.Coyote.Specifications/Monitor.State.OnEventDoActionAttribute/
[285]: ref/Microsoft.Coyote.Specifications/Monitor.State.OnEventDoActionAttribute/OnEventDoActionAtt
ribute/
[286]: ref/Microsoft.Coyote.Specifications/Monitor.State.IgnoreEventsAttribute/
[287]: ref/Microsoft.Coyote.Specifications/Monitor.State.IgnoreEventsAttribute/IgnoreEventsAttribute
/
[288]: ref/Microsoft.Coyote.Specifications/Monitor.State.ColdAttribute/
[289]: ref/Microsoft.Coyote.Specifications/Monitor.State.ColdAttribute/ColdAttribute/
[290]: ref/Microsoft.Coyote.Specifications/Monitor.State.HotAttribute/
[291]: ref/Microsoft.Coyote.Specifications/Monitor.State.HotAttribute/HotAttribute/
[292]: ref/Microsoft.Coyote.Actors/
[293]: ref/Microsoft.Coyote.ActorsNamespace/
[294]: ref/Microsoft.Coyote.Actors/DequeueStatus/
[295]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/
[296]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/SetResult/
[297]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/TrySetResult/
[298]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/SetCancelled/
[299]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/TrySetCanceled/
[300]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/SetException/
[301]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/TrySetException/
[302]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/GetAwaiter/
[303]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/AwaitableEventGroup/
[304]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/Task/
[305]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/IsCompleted/
[306]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/IsCanceled/
[307]: ref/Microsoft.Coyote.Actors/AwaitableEventGroup-1/IsFaulted/
[308]: ref/Microsoft.Coyote.Actors/DefaultEvent/
[309]: ref/Microsoft.Coyote.Actors/DefaultEvent/Instance/
[310]: ref/Microsoft.Coyote.Actors/Event/
[311]: ref/Microsoft.Coyote.Actors/Event/Event/
[312]: ref/Microsoft.Coyote.Actors/EventGroup/
[313]: ref/Microsoft.Coyote.Actors/EventGroup/EventGroup/
[314]: ref/Microsoft.Coyote.Actors/EventGroup/Id/
[315]: ref/Microsoft.Coyote.Actors/EventGroup/Name/
[316]: ref/Microsoft.Coyote.Actors/HaltEvent/
[317]: ref/Microsoft.Coyote.Actors/HaltEvent/Instance/
[318]: ref/Microsoft.Coyote.Actors/WildCardEvent/
[319]: ref/Microsoft.Coyote.Actors/WildCardEvent/WildCardEvent/
[320]: ref/Microsoft.Coyote.Actors/OnExceptionOutcome/
[321]: ref/Microsoft.Coyote.Actors/UnhandledEventException/
[322]: ref/Microsoft.Coyote.Actors/UnhandledEventException/UnhandledEvent/
[323]: ref/Microsoft.Coyote.Actors/UnhandledEventException/CurrentStateName/
[324]: ref/Microsoft.Coyote.Actors/OnActorHaltedHandler/
[325]: ref/Microsoft.Coyote.Actors/OnEventDroppedHandler/
[326]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/
[327]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnCreateActor/
[328]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnCreateStateMachine/
[329]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnCreateTimer/
[330]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnDefaultEventHandler/
[331]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnEventHandlerTerminated/
[332]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnDequeueEvent/
[333]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnEnqueueEvent/
[334]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnExceptionHandled/
[335]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnExceptionThrown/
[336]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnExecuteAction/
[337]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnGotoState/
[338]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnHalt/
[339]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnPopState/
[340]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnPopStateUnhandledEvent/
[341]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnPushState/
[342]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnRaiseEvent/
[343]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnHandleRaisedEvent/
[344]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnReceiveEvent/
[345]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnSendEvent/
[346]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnStateTransition/
[347]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnStopTimer/
[348]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/OnWaitEvent/
[349]: ref/Microsoft.Coyote.Actors/ActorRuntimeLogTextFormatter/ActorRuntimeLogTextFormatter/
[350]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/
[351]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnCreateActor/
[352]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnCreateStateMachine/
[353]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnExecuteAction/
[354]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnSendEvent/
[355]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnRaiseEvent/
[356]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnHandleRaisedEvent/
[357]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnEnqueueEvent/
[358]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnDequeueEvent/
[359]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnReceiveEvent/
[360]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnWaitEvent/
[361]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnStateTransition/
[362]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnGotoState/
[363]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnPushState/
[364]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnPopState/
[365]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnDefaultEventHandler/
[366]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnEventHandlerTerminated/
[367]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnHalt/
[368]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnPopStateUnhandledEvent/
[369]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnExceptionThrown/
[370]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnExceptionHandled/
[371]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnCreateTimer/
[372]: ref/Microsoft.Coyote.Actors/IActorRuntimeLog/OnStopTimer/
[373]: ref/Microsoft.Coyote.Actors/Actor/
[374]: ref/Microsoft.Coyote.Actors/Actor/CreateActor/
[375]: ref/Microsoft.Coyote.Actors/Actor/SendEvent/
[376]: ref/Microsoft.Coyote.Actors/Actor/ReceiveEventAsync/
[377]: ref/Microsoft.Coyote.Actors/Actor/StartTimer/
[378]: ref/Microsoft.Coyote.Actors/Actor/StartPeriodicTimer/
[379]: ref/Microsoft.Coyote.Actors/Actor/StopTimer/
[380]: ref/Microsoft.Coyote.Actors/Actor/RandomBoolean/
[381]: ref/Microsoft.Coyote.Actors/Actor/RandomInteger/
[382]: ref/Microsoft.Coyote.Actors/Actor/Monitor/
[383]: ref/Microsoft.Coyote.Actors/Actor/Assert/
[384]: ref/Microsoft.Coyote.Actors/Actor/RaiseHaltEvent/
[385]: ref/Microsoft.Coyote.Actors/Actor/OnInitializeAsync/
[386]: ref/Microsoft.Coyote.Actors/Actor/OnEventDequeuedAsync/
[387]: ref/Microsoft.Coyote.Actors/Actor/OnEventIgnored/
[388]: ref/Microsoft.Coyote.Actors/Actor/OnEventDeferred/
[389]: ref/Microsoft.Coyote.Actors/Actor/OnEventHandledAsync/
[390]: ref/Microsoft.Coyote.Actors/Actor/OnEventUnhandledAsync/
[391]: ref/Microsoft.Coyote.Actors/Actor/OnExceptionHandledAsync/
[392]: ref/Microsoft.Coyote.Actors/Actor/OnHaltAsync/
[393]: ref/Microsoft.Coyote.Actors/Actor/OnException/
[394]: ref/Microsoft.Coyote.Actors/Actor/Equals/
[395]: ref/Microsoft.Coyote.Actors/Actor/GetHashCode/
[396]: ref/Microsoft.Coyote.Actors/Actor/ToString/
[397]: ref/Microsoft.Coyote.Actors/Actor/Actor/
[398]: ref/Microsoft.Coyote.Actors/Actor/Id/
[399]: ref/Microsoft.Coyote.Actors/Actor/CurrentEventGroup/
[400]: ref/Microsoft.Coyote.Actors/Actor/Logger/
[401]: ref/Microsoft.Coyote.Actors/Actor/HashedState/
[402]: ref/Microsoft.Coyote.Actors/ActorExecutionStatus/
[403]: ref/Microsoft.Coyote.Actors/ActorId/
[404]: ref/Microsoft.Coyote.Actors/ActorId/Equals/
[405]: ref/Microsoft.Coyote.Actors/ActorId/GetHashCode/
[406]: ref/Microsoft.Coyote.Actors/ActorId/ToString/
[407]: ref/Microsoft.Coyote.Actors/ActorId/CompareTo/
[408]: ref/Microsoft.Coyote.Actors/ActorId/Runtime/
[409]: ref/Microsoft.Coyote.Actors/ActorId/IsNameUsedForHashing/
[410]: ref/Microsoft.Coyote.Actors/ActorId/Value/
[411]: ref/Microsoft.Coyote.Actors/ActorId/NameValue/
[412]: ref/Microsoft.Coyote.Actors/ActorId/Type/
[413]: ref/Microsoft.Coyote.Actors/ActorId/Name/
[414]: ref/Microsoft.Coyote.Actors/IActorRuntime/
[415]: ref/Microsoft.Coyote.Actors/IActorRuntime/CreateActorId/
[416]: ref/Microsoft.Coyote.Actors/IActorRuntime/CreateActorIdFromName/
[417]: ref/Microsoft.Coyote.Actors/IActorRuntime/CreateActor/
[418]: ref/Microsoft.Coyote.Actors/IActorRuntime/SendEvent/
[419]: ref/Microsoft.Coyote.Actors/IActorRuntime/GetCurrentEventGroup/
[420]: ref/Microsoft.Coyote.Actors/IActorRuntime/GetActorExecutionStatus/
[421]: ref/Microsoft.Coyote.Actors/IActorRuntime/GetCurrentActorIds/
[422]: ref/Microsoft.Coyote.Actors/IActorRuntime/GetCurrentActorTypes/
[423]: ref/Microsoft.Coyote.Actors/IActorRuntime/GetCurrentActorCount/
[424]: ref/Microsoft.Coyote.Actors/IActorRuntime/OnActorHalted/
[425]: ref/Microsoft.Coyote.Actors/IActorRuntime/OnEventDropped/
[426]: ref/Microsoft.Coyote.Actors/RuntimeFactory/
[427]: ref/Microsoft.Coyote.Actors/RuntimeFactory/Create/
[428]: ref/Microsoft.Coyote.Actors/SendOptions/
[429]: ref/Microsoft.Coyote.Actors/SendOptions/ToString/
[430]: ref/Microsoft.Coyote.Actors/SendOptions/SendOptions/
[431]: ref/Microsoft.Coyote.Actors/SendOptions/Default/
[432]: ref/Microsoft.Coyote.Actors/SendOptions/MustHandle/
[433]: ref/Microsoft.Coyote.Actors/SendOptions/Assert/
[434]: ref/Microsoft.Coyote.Actors/SendOptions/HashedState/
[435]: ref/Microsoft.Coyote.Actors/StateMachine/
[436]: ref/Microsoft.Coyote.Actors/StateMachine/RaiseEvent/
[437]: ref/Microsoft.Coyote.Actors/StateMachine/RaiseGotoStateEvent/
[438]: ref/Microsoft.Coyote.Actors/StateMachine/RaisePushStateEvent/
[439]: ref/Microsoft.Coyote.Actors/StateMachine/RaisePopStateEvent/
[440]: ref/Microsoft.Coyote.Actors/StateMachine/RaiseHaltEvent/
[441]: ref/Microsoft.Coyote.Actors/StateMachine/OnEventHandledAsync/
[442]: ref/Microsoft.Coyote.Actors/StateMachine/StateMachine/
[443]: ref/Microsoft.Coyote.Actors/StateMachine/CurrentState/
[444]: ref/Microsoft.Coyote.Actors/Actor.OnEventDoActionAttribute/
[445]: ref/Microsoft.Coyote.Actors/Actor.OnEventDoActionAttribute/OnEventDoActionAttribute/
[446]: ref/Microsoft.Coyote.Actors/StateMachine.State/
[447]: ref/Microsoft.Coyote.Actors/StateMachine.State/State/
[448]: ref/Microsoft.Coyote.Actors/StateMachine.StateGroup/
[449]: ref/Microsoft.Coyote.Actors/StateMachine.StateGroup/StateGroup/
[450]: ref/Microsoft.Coyote.Actors/StateMachine.State.StartAttribute/
[451]: ref/Microsoft.Coyote.Actors/StateMachine.State.StartAttribute/StartAttribute/
[452]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnEntryAttribute/
[453]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnEntryAttribute/OnEntryAttribute/
[454]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnExitAttribute/
[455]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnExitAttribute/OnExitAttribute/
[456]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnEventGotoStateAttribute/
[457]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnEventGotoStateAttribute/OnEventGotoStateAttr
ibute/
[458]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnEventPushStateAttribute/
[459]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnEventPushStateAttribute/OnEventPushStateAttr
ibute/
[460]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnEventDoActionAttribute/
[461]: ref/Microsoft.Coyote.Actors/StateMachine.State.OnEventDoActionAttribute/OnEventDoActionAttrib
ute/
[462]: ref/Microsoft.Coyote.Actors/StateMachine.State.DeferEventsAttribute/
[463]: ref/Microsoft.Coyote.Actors/StateMachine.State.DeferEventsAttribute/DeferEventsAttribute/
[464]: ref/Microsoft.Coyote.Actors/StateMachine.State.IgnoreEventsAttribute/
[465]: ref/Microsoft.Coyote.Actors/StateMachine.State.IgnoreEventsAttribute/IgnoreEventsAttribute/
[466]: ref/Microsoft.Coyote.Actors.CoverageNamespace/
[467]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo.md
[468]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo/IsMachineDeclared.md
[469]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo/DeclareMachineState.md
[470]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo/DeclareMachineStateEventPair.md
[471]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo/Merge.md
[472]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo/ActorCoverageInfo.md
[473]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo/Machines.md
[474]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo/MachinesToStates.md
[475]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo/RegisteredActorEvents.md
[476]: ref/Microsoft.Coyote.Actors.Coverage/ActorCoverageInfo/ActorEventInfo.md
[477]: ref/Microsoft.Coyote.Actors.Coverage/ActorEventCoverage.md
[478]: ref/Microsoft.Coyote.Actors.Coverage/ActorEventCoverage/GetEventsReceived.md
[479]: ref/Microsoft.Coyote.Actors.Coverage/ActorEventCoverage/GetEventsSent.md
[480]: ref/Microsoft.Coyote.Actors.Coverage/ActorEventCoverage/ActorEventCoverage.md
[481]: ref/Microsoft.Coyote.Actors.SharedObjectsNamespace/
[482]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedCounter/
[483]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedCounter/Create/
[484]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedCounter/Increment/
[485]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedCounter/Decrement/
[486]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedCounter/GetValue/
[487]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedCounter/Add/
[488]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedCounter/Exchange/
[489]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedCounter/CompareExchange/
[490]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedDictionary/
[491]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedDictionary/Create/
[492]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedDictionary-2/
[493]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedDictionary-2/TryAdd/
[494]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedDictionary-2/TryUpdate/
[495]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedDictionary-2/TryGetValue/
[496]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedDictionary-2/TryRemove/
[497]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedDictionary-2/Item/
[498]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedDictionary-2/Count/
[499]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedRegister/
[500]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedRegister/Create/
[501]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedRegister-1/
[502]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedRegister-1/Update/
[503]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedRegister-1/GetValue/
[504]: ref/Microsoft.Coyote.Actors.SharedObjects/SharedRegister-1/SetValue/
[505]: ref/Microsoft.Coyote.Actors.TimersNamespace/
[506]: ref/Microsoft.Coyote.Actors.Timers/TimerElapsedEvent/
[507]: ref/Microsoft.Coyote.Actors.Timers/TimerElapsedEvent/TimerElapsedEvent/
[508]: ref/Microsoft.Coyote.Actors.Timers/TimerElapsedEvent/Info/
[509]: ref/Microsoft.Coyote.Actors.Timers/TimerInfo/
[510]: ref/Microsoft.Coyote.Actors.Timers/TimerInfo/Equals/
[511]: ref/Microsoft.Coyote.Actors.Timers/TimerInfo/GetHashCode/
[512]: ref/Microsoft.Coyote.Actors.Timers/TimerInfo/ToString/
[513]: ref/Microsoft.Coyote.Actors.Timers/TimerInfo/OwnerId/
[514]: ref/Microsoft.Coyote.Actors.Timers/TimerInfo/DueTime/
[515]: ref/Microsoft.Coyote.Actors.Timers/TimerInfo/Period/
[516]: ref/Microsoft.Coyote.Actors.Timers/TimerInfo/CustomEvent/
[517]: ref/Microsoft.Coyote.Actors.UnitTestingNamespace/
[518]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/
[519]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/StartActorAsync/
[520]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/SendEventAsync/
[521]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/Invoke/
[522]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/InvokeAsync/
[523]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/Assert/
[524]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/AssertStateTransition/
[525]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/AssertIsWaitingToReceiveEvent/
[526]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/AssertInboxSize/
[527]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/ActorTestKit/
[528]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/Logger/
[529]: ref/Microsoft.Coyote.Actors.UnitTesting/ActorTestKit-1/ActorInstance/
[530]: ref/Microsoft.Coyote.Test/
[531]: ref/Microsoft.Coyote.RewritingNamespace/
[532]: ref/Microsoft.Coyote.Rewriting/RewritingSignatureAttribute/
[533]: ref/Microsoft.Coyote.Rewriting/RewritingSignatureAttribute/RewritingSignatureAttribute/
[534]: ref/Microsoft.Coyote.Rewriting/RewritingSignatureAttribute/Version/
[535]: ref/Microsoft.Coyote.Rewriting/RewritingSignatureAttribute/Signature/
[536]: ref/Microsoft.Coyote.Rewriting/SkipRewritingAttribute/
[537]: ref/Microsoft.Coyote.Rewriting/SkipRewritingAttribute/SkipRewritingAttribute/
[538]: ref/Microsoft.Coyote.Rewriting/SkipRewritingAttribute/Reason/
[539]: ref/Microsoft.Coyote.Rewriting/RewritingEngine/
[540]: ref/Microsoft.Coyote.Rewriting/RewritingEngine/IsAssemblyRewritten/
[541]: ref/Microsoft.Coyote.SystematicTestingNamespace/
[542]: ref/Microsoft.Coyote.SystematicTesting/TestReport/
[543]: ref/Microsoft.Coyote.SystematicTesting/TestReport/Merge/
[544]: ref/Microsoft.Coyote.SystematicTesting/TestReport/GetText/
[545]: ref/Microsoft.Coyote.SystematicTesting/TestReport/Clone/
[546]: ref/Microsoft.Coyote.SystematicTesting/TestReport/TestReport/
[547]: ref/Microsoft.Coyote.SystematicTesting/TestReport/Configuration/
[548]: ref/Microsoft.Coyote.SystematicTesting/TestReport/CoverageInfo/
[549]: ref/Microsoft.Coyote.SystematicTesting/TestReport/NumOfExploredFairPaths/
[550]: ref/Microsoft.Coyote.SystematicTesting/TestReport/NumOfExploredUnfairPaths/
[551]: ref/Microsoft.Coyote.SystematicTesting/TestReport/NumOfFoundBugs/
[552]: ref/Microsoft.Coyote.SystematicTesting/TestReport/BugReports/
[553]: ref/Microsoft.Coyote.SystematicTesting/TestReport/UncontrolledInvocations/
[554]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MinControlledOperations/
[555]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MaxControlledOperations/
[556]: ref/Microsoft.Coyote.SystematicTesting/TestReport/TotalControlledOperations/
[557]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MinConcurrencyDegree/
[558]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MaxConcurrencyDegree/
[559]: ref/Microsoft.Coyote.SystematicTesting/TestReport/TotalConcurrencyDegree/
[560]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MinOperationGroupingDegree/
[561]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MaxOperationGroupingDegree/
[562]: ref/Microsoft.Coyote.SystematicTesting/TestReport/TotalOperationGroupingDegree/
[563]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MinExploredFairSteps/
[564]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MaxExploredFairSteps/
[565]: ref/Microsoft.Coyote.SystematicTesting/TestReport/TotalExploredFairSteps/
[566]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MinExploredUnfairSteps/
[567]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MaxExploredUnfairSteps/
[568]: ref/Microsoft.Coyote.SystematicTesting/TestReport/TotalExploredUnfairSteps/
[569]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MaxFairStepsHitInFairTests/
[570]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MaxUnfairStepsHitInFairTests/
[571]: ref/Microsoft.Coyote.SystematicTesting/TestReport/MaxUnfairStepsHitInUnfairTests/
[572]: ref/Microsoft.Coyote.SystematicTesting/TestReport/InternalErrors/
[573]: ref/Microsoft.Coyote.SystematicTesting/TestAttribute/
[574]: ref/Microsoft.Coyote.SystematicTesting/TestAttribute/TestAttribute/
[575]: ref/Microsoft.Coyote.SystematicTesting/TestInitAttribute/
[576]: ref/Microsoft.Coyote.SystematicTesting/TestInitAttribute/TestInitAttribute/
[577]: ref/Microsoft.Coyote.SystematicTesting/TestDisposeAttribute/
[578]: ref/Microsoft.Coyote.SystematicTesting/TestDisposeAttribute/TestDisposeAttribute/
[579]: ref/Microsoft.Coyote.SystematicTesting/TestIterationDisposeAttribute/
[580]: ref/Microsoft.Coyote.SystematicTesting/TestIterationDisposeAttribute/TestIterationDisposeAttr
ibute/
[581]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/
[582]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/Create/
[583]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/Run/
[584]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/Stop/
[585]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/GetReport/
[586]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/ThrowIfBugFound/
[587]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/TryEmitReports/
[588]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/TryEmitCoverageReports/
[589]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/RegisterStartIterationCallBack/
[590]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/RegisterEndIterationCallBack/
[591]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/InvokeStartIterationCallBacks/
[592]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/InvokeEndIterationCallBacks/
[593]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/IsTestRewritten/
[594]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/SetLogger/
[595]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/Dispose/
[596]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/TestReport/
[597]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/ReadableTrace/
[598]: ref/Microsoft.Coyote.SystematicTesting/TestingEngine/ReproducibleTrace/
[599]: ref/Microsoft.Coyote.SystematicTesting.Frameworks.XUnitNamespace/
[600]: ref/Microsoft.Coyote.SystematicTesting.Frameworks.XUnit/TestOutputLogger/
[601]: ref/Microsoft.Coyote.SystematicTesting.Frameworks.XUnit/TestOutputLogger/Write/
[602]: ref/Microsoft.Coyote.SystematicTesting.Frameworks.XUnit/TestOutputLogger/WriteLine/
[603]: ref/Microsoft.Coyote.SystematicTesting.Frameworks.XUnit/TestOutputLogger/Dispose/
[604]: ref/Microsoft.Coyote.SystematicTesting.Frameworks.XUnit/TestOutputLogger/TestOutputLogger/
[605]: ref/Microsoft.Coyote.WebNamespace/
[606]: ref/Microsoft.Coyote.Web/RequestControllerMiddlewareExtensions/
[607]: ref/Microsoft.Coyote.Web/RequestControllerMiddlewareExtensions/UseRequestController/
