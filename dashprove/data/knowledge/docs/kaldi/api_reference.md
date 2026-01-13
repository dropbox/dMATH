───────────┬───────┬──
[[Logo]][1]│Kaldi  │  
───────────┴───────┴──
Class Index
[a][2] | [b][3] | [c][4] | [d][5] | [e][6] | [f][7] | [g][8] | [h][9] | [i][10] | [k][11] | [l][12]
| [m][13] | [n][14] | [o][15] | [p][16] | [q][17] | [r][18] | [s][19] | [t][20] | [u][21] | [v][22]
| [w][23] | [x][24]

─────────────────────┬─────────────────┬──────────────────────┬──────────────────┬──────────────────
a                    │[DecodableAmNnetS│[KwsAlignment][27]    │[NormalizeComponen│[Sgmm2GselectConfi
                     │impleLooped][25] │([kaldi][28])         │t][29]            │g][31]            
                     │([kaldi::nnet3][2│                      │([kaldi::nnet2][30│([kaldi][32])     
                     │6])              │                      │])                │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[DecodableAmNnetSimpl│[KwScoreStats][35│[NumberIstream][37]   │[Sgmm2LikelihoodCa
eParallel][33]       │]                │([kaldi][38])         │che][39]          
([kaldi::nnet3][34]) │([kaldi::kws_inte│                      │([kaldi][40])     
                     │rnal][36])       │                      │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[Access][41]         │[DecodableAmSgmm2│[KwsProductFstToKwsLex│o                 │[Sgmm2PerFrameDeri
([kaldi::nnet3][42]) │][43]            │icographicFstMapper][4│                  │vedVars][47]      
                     │([kaldi][44])    │5] ([kaldi][46])      │                  │([kaldi][48])     
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[AccumAmDiagGmm][49] │[DecodableAmSgmm2│[KwsTerm][53]         │[Sgmm2PerSpkDerive
([kaldi][50])        │Scaled][51]      │([kaldi][54])         │dVars][55]        
                     │([kaldi][52])    │                      │([kaldi][56])     
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[AccumDiagGmm][57]   │[DecodableDiagGmm│[KwsTermsAligner][61] │[ObjectiveFunction│[Sgmm2Project][65]
([kaldi][58])        │ScaledOnline][59]│([kaldi][62])         │Info][63]         │([kaldi][66])     
                     │([kaldi][60])    │                      │([kaldi::nnet3][64│                  
                     │                 │                      │])                │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AccumFullGmm][67]   │[DecodableInterfa│[KwsTermsAlignerOption│[OfflineFeatureTpl│[Sgmm2SplitSubstat
([kaldi][68])        │ce][69]          │s][71] ([kaldi][72])  │][73]             │esConfig][75]     
                     │([kaldi][70])    │                      │([kaldi][74])     │([kaldi][76])     
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AccumulateMultiThrea│[DecodableMapped]│[KwTermEqual][81]     │[ConvolutionModel:│[ShiftedDeltaFeatu
dedClass][77]        │[79]             │([kaldi::kws_internal]│:Offset][83]      │res][85]          
([kaldi][78])        │([kaldi][80])    │[82])                 │([kaldi::nnet3::ti│([kaldi][86])     
                     │                 │                      │me_height_convolut│                  
                     │                 │                      │ion][84])         │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AccumulateTreeStatsI│[DecodableMatrixM│[KwTermLower][91]     │[OffsetFileInputIm│[ShiftedDeltaFeatu
nfo][87]             │apped][89]       │([kaldi::kws_internal]│pl][93]           │resOptions][95]   
([kaldi][88])        │([kaldi][90])    │[92])                 │([kaldi][94])     │([kaldi][96])     
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AccumulateTreeStatsO│[DecodableMatrixM│l                     │[OffsetForwardingD│[Sigmoid][103]    
ptions][97]          │appedOffset][99] │                      │escriptor][101]   │([kaldi::nnet1][10
([kaldi][98])        │([kaldi][100])   │                      │([kaldi::nnet3][10│4])               
                     │                 │                      │2])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[ActivePath][105]    │[DecodableMatrixS│[OnlineAppendFeature][│[SigmoidComponent]
([kaldi][106])       │caled][107]      │109] ([kaldi][110])   │[111]             
                     │([kaldi][108])   │                      │([kaldi::nnet2][11
                     │                 │                      │2])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[AdditiveNoiseCompone│[DecodableMatrixS│[LatticeArcRecord][117│[OnlineAudioSource│[SigmoidComponent]
nt][113]             │caledMapped][115]│] ([kaldi][118])      │Itf][119]         │[121]             
([kaldi::nnet2][114])│([kaldi][116])   │                      │([kaldi][120])    │([kaldi::nnet3][12
                     │                 │                      │                  │2])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AddShift][123]      │[DecodableNnet2On│[LatticeBiglmFasterDec│[OnlineBaseFeature│[SimpleDecoder][13
([kaldi::nnet1][124])│line][125]       │oder][127]            │][129]            │1] ([kaldi][132]) 
                     │([kaldi::nnet2][1│([kaldi][128])        │([kaldi][130])    │                  
                     │26])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AffineComponent][133│[DecodableNnet2On│[LatticeDeterminizer][│[OnlineCacheFeatur│[SimpleForwardingD
]                    │lineOptions][135]│137] ([fst][138])     │e][139]           │escriptor][141]   
([kaldi::nnet2][134])│([kaldi::nnet2][1│                      │([kaldi][140])    │([kaldi::nnet3][14
                     │36])             │                      │                  │2])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AffineComponent][143│[DecodableNnetLoo│[LatticeDeterminizerPr│[OnlineCacheInput]│[SimpleMeanTransfo
]                    │pedOnline][145]  │uned][147]            │[149]             │rm][151]          
([kaldi::nnet3][144])│([kaldi::nnet3][1│([fst][148])          │([kaldi][150])    │([kaldi::different
                     │46])             │                      │                  │iable_transform][1
                     │                 │                      │                  │52])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AffineComponentPreco│[DecodableNnetLoo│[LatticeFasterDecoderC│[OnlineCmnInput][1│[SimpleObjectiveIn
nditioned][153]      │pedOnlineBase][15│onfig][157]           │59] ([kaldi][160])│fo][161]          
([kaldi::nnet2][154])│5]               │([kaldi][158])        │                  │([kaldi::nnet3][16
                     │([kaldi::nnet3][1│                      │                  │2])               
                     │56])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AffineComponentPreco│[DecodableNnetSim│[LatticeFasterDecoderT│[OnlineCmvn][169] │[SimpleOptions][17
nditionedOnline][163]│ple][165]        │pl][167]              │([kaldi][170])    │1] ([kaldi][172]) 
([kaldi::nnet2][164])│([kaldi::nnet3][1│([kaldi][168])        │                  │                  
                     │66])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AffineTransform][173│[DecodableNnetSim│[LatticeFasterOnlineDe│[OnlineCmvnOptions│[SimpleSentenceAve
]                    │pleLooped][175]  │coderTpl][177]        │][179]            │ragingComponent][1
([kaldi::nnet1][174])│([kaldi::nnet3][1│([kaldi][178])        │([kaldi][180])    │81]               
                     │76])             │                      │                  │([kaldi::nnet1][18
                     │                 │                      │                  │2])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AffineXformStats][18│[DecodableNnetSim│[LatticeHolder][187]  │[OnlineCmvnState][│[SimpleSumDescript
3] ([kaldi][184])    │pleLoopedInfo][18│([kaldi][188])        │189]              │or][191]          
                     │5]               │                      │([kaldi][190])    │([kaldi::nnet3][19
                     │([kaldi::nnet3][1│                      │                  │2])               
                     │86])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AgglomerativeCluster│[DecodableSum][19│[LatticeIncrementalDec│[OnlineDecodableDi│[FmllrRawAccs::Sin
er][193]             │5] ([kaldi][196])│oderConfig][197]      │agGmmScaled][199] │gleFrameStats][201
([kaldi][194])       │                 │([kaldi][198])        │([kaldi][200])    │] ([kaldi][202])  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AhcCluster][203]    │[DecodableSumScal│[LatticeIncrementalDec│[OnlineDeltaFeatur│[FmllrDiagGmmAccs:
([kaldi][204])       │ed][205]         │oderTpl][207]         │e][209]           │:SingleFrameStats]
                     │([kaldi][206])   │([kaldi][208])        │([kaldi][210])    │[211]             
                     │                 │                      │                  │([kaldi][212])    
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AlignConfig][213]   │[DecodeInfo][215]│[LatticeIncrementalDet│[OnlineDeltaInput]│[RowOpsSplitter::S
([kaldi][214])       │                 │erminizer][216]       │[218]             │ingleSplitInfo][22
                     │                 │([kaldi][217])        │([kaldi][219])    │0]                
                     │                 │                      │                  │([kaldi::nnet3][22
                     │                 │                      │                  │1])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AlignedTermsPair][22│[DecodeUtteranceL│[LatticeIncrementalOnl│[OnlineEndpointCon│[SingleUtteranceGm
2] ([kaldi][223])    │atticeFasterClass│ineDecoderTpl][226]   │fig][228]         │mDecoder][230]    
                     │][224]           │([kaldi][227])        │([kaldi][229])    │([kaldi][231])    
                     │([kaldi][225])   │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AmDiagGmm][232]     │[DeltaFeatures][2│[DiscriminativeSupervi│[OnlineEndpointRul│[SingleUtteranceNn
([kaldi][233])       │34]              │sionSplitter::LatticeI│e][238]           │et2Decoder][240]  
                     │([kaldi][235])   │nfo][236]             │([kaldi][239])    │([kaldi][241])    
                     │                 │([kaldi::discriminativ│                  │                  
                     │                 │e][237])              │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AmNnet][242]        │[DeltaFeaturesOpt│[LatticeLexiconWordAli│[OnlineFasterDecod│[SingleUtteranceNn
([kaldi::nnet2][243])│ions][244]       │gner][246]            │er][248]          │et2DecoderThreaded
                     │([kaldi][245])   │([kaldi][247])        │([kaldi][249])    │][250]            
                     │                 │                      │                  │([kaldi][251])    
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AmNnetSimple][252]  │[DerivativeTimeLi│[LatticePhoneAligner][│[OnlineFasterDecod│[SingleUtteranceNn
([kaldi::nnet3][253])│miter][254]      │256] ([kaldi][257])   │erOpts][258]      │et3DecoderTpl][260
                     │([kaldi::nnet3][2│                      │([kaldi][259])    │] ([kaldi][261])  
                     │55])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AmSgmm2][262]       │[Descriptor][264]│[LatticeReader][266]  │[OnlineFeatInputIt│[SingleUtteranceNn
([kaldi][263])       │([kaldi::nnet3][2│([kaldi][267])        │f][268]           │et3IncrementalDeco
                     │65])             │                      │([kaldi][269])    │derTpl][270]      
                     │                 │                      │                  │([kaldi][271])    
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[Analyzer][272]      │[DeterministicOnD│[LatticeSimpleDecoder]│[OnlineFeatureInte│[SlidingWindowCmnO
([kaldi::nnet3][273])│emandFst][274]   │[276] ([kaldi][277])  │rface][278]       │ptions][280]      
                     │([fst][275])     │                      │([kaldi][279])    │([kaldi][281])    
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AppendTransform][282│[DeterminizeLatti│[LatticeSimpleDecoderC│[OnlineFeatureMatr│[SoftHingeComponen
]                    │ceOptions][284]  │onfig][286]           │ix][288]          │t][290]           
([kaldi::differentiab│([fst][285])     │([kaldi][287])        │([kaldi][289])    │([kaldi::nnet2][29
le_transform][283])  │                 │                      │                  │1])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArbitraryResample][2│[DeterminizeLatti│[PrunedCompactLatticeC│[OnlineFeatureMatr│[Softmax][300]    
92] ([kaldi][293])   │cePhonePrunedOpti│omposer::LatticeStateI│ixOptions][298]   │([kaldi::nnet1][30
                     │ons][294]        │nfo][296]             │([kaldi][299])    │1])               
                     │([fst][295])     │([kaldi][297])        │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[MinimumBayesRisk::Ar│[DeterminizeLatti│[LatticeStringReposito│[OnlineFeaturePipe│[SoftmaxComponent]
c][302]              │cePrunedOptions][│ry][306] ([fst][307]) │line][308]        │[310]             
([kaldi][303])       │304] ([fst][305])│                      │([kaldi][309])    │([kaldi::nnet2][31
                     │                 │                      │                  │1])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[GrammarFstPreparer::│[DeterminizeLatti│[LatticeToStdMapper][3│[OnlineFeaturePipe│[SoftmaxComponent]
ArcCategory][312]    │ceTask][314]     │16] ([fst][317])      │lineCommandLineCon│[320]             
([fst][313])         │([kaldi][315])   │                      │fig][318]         │([kaldi::nnet3][32
                     │                 │                      │([kaldi][319])    │1])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArcIterator<        │[DeterminizerStar│[LatticeWeightTpl][326│[OnlineFeaturePipe│[SolverOptions][33
GrammarFst >][322]   │][324]           │] ([fst][327])        │lineConfig][328]  │0] ([kaldi][331]) 
([fst][323])         │([fst][325])     │                      │([kaldi][329])    │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArcIterator<        │[DfsOrderVisitor]│[LatticeWordAligner][3│[OnlineFeInput][33│[SparseMatrix][340
TrivialFactorWeightFs│[334]            │36] ([kaldi][337])    │8] ([kaldi][339]) │] ([kaldi][341])  
t< A, F > >][332]    │([fst][335])     │                      │                  │                  
([fst][333])         │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArcPosteriorComputer│[DiagGmm][344]   │[LbfgsOptions][346]   │[OnlineGenericBase│[SparseVector][350
][342] ([kaldi][343])│([kaldi][345])   │([kaldi][347])        │Feature][348]     │] ([kaldi][351])  
                     │                 │                      │([kaldi][349])    │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArcticWeightTpl][352│[DiagGmmNormal][3│[LdaEstimate][356]    │[OnlineGmmAdaptati│[FmllrTransform::S
] ([fst][353])       │54]              │([kaldi][357])        │onState][358]     │peakerStats][360] 
                     │([kaldi][355])   │                      │([kaldi][359])    │([kaldi::different
                     │                 │                      │                  │iable_transform][3
                     │                 │                      │                  │61])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArpaFileParser][362]│[DifferentiableTr│[LdaEstimateOptions][3│[OnlineGmmDecoding│[SpeakerStatsItf][
([kaldi][363])       │ansform][364]    │66] ([kaldi][367])    │AdaptationPolicyCo│370]              
                     │([kaldi::differen│                      │nfig][368]        │([kaldi::different
                     │tiable_transform]│                      │([kaldi][369])    │iable_transform][3
                     │[365])           │                      │                  │71])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArpaLine][372]      │[DiscriminativeCo│[LengthNormComponent][│[OnlineGmmDecoding│[SpecAugmentTimeMa
([kaldi][373])       │mputation][374]  │376]                  │Config][378]      │skComponent][380] 
                     │([kaldi::discrimi│([kaldi::nnet1][377]) │([kaldi][379])    │([kaldi::nnet3][38
                     │native][375])    │                      │                  │1])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArpaLmCompiler][382]│[DiscriminativeEx│[DecodableAmDiagGmmUnm│[OnlineGmmDecoding│[SpecAugmentTimeMa
([kaldi][383])       │ampleMerger][384]│apped::LikelihoodCache│Models][388]      │skComponentPrecomp
                     │([kaldi::nnet3][3│Record][386]          │([kaldi][389])    │utedIndexes][390] 
                     │85])             │([kaldi][387])        │                  │([kaldi::nnet3][39
                     │                 │                      │                  │1])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArpaLmCompilerImpl][│[DiscriminativeEx│[LimitRankClass][396] │[OnlineIvectorEsti│[SpectrogramComput
392] ([kaldi][393])  │ampleSplitter][39│([kaldi::nnet2][397]) │mationStats][398] │er][400]          
                     │4]               │                      │([kaldi][399])    │([kaldi][401])    
                     │([kaldi::nnet2][3│                      │                  │                  
                     │95])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArpaLmCompilerImplIn│[DiscriminativeEx│[LinearCgdOptions][406│[OnlineIvectorExtr│[SpectrogramOption
terface][402]        │amplesRepository]│] ([kaldi][407])      │actionConfig][408]│s][410]           
([kaldi][403])       │[404]            │                      │([kaldi][409])    │([kaldi][411])    
                     │([kaldi::nnet2][4│                      │                  │                  
                     │05])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ArpaParseOptions][41│[DiscriminativeNn│[LinearComponent][416]│[OnlineIvectorExtr│[SpeexOptions][420
2] ([kaldi][413])    │etExample][414]  │([kaldi::nnet3][417]) │actionInfo][418]  │] ([kaldi][421])  
                     │([kaldi::nnet2][4│                      │([kaldi][419])    │                  
                     │15])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[AveragePoolingCompon│[DiscriminativeOb│[LinearResample][426] │[OnlineIvectorExtr│[SphinxMatrixHolde
ent][422]            │jectiveFunctionIn│([kaldi][427])        │actorAdaptationSta│r][430]           
([kaldi::nnet1][423])│fo][424]         │                      │te][428]          │([kaldi][431])    
                     │([kaldi::nnet3][4│                      │([kaldi][429])    │                  
                     │25])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
b                    │[DiscriminativeOb│[LinearTransform][434]│[OnlineIvectorFeat│[Splice][438]     
                     │jectiveInfo][432]│([kaldi::nnet1][435]) │ure][436]         │([kaldi::nnet1][43
                     │([kaldi::discrimi│                      │([kaldi][437])    │9])               
                     │native][433])    │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[DiscriminativeOption│[LinearVtln][442]│[OnlineLdaInput][444] │[SpliceComponent][
s][440]              │([kaldi][443])   │([kaldi][445])        │446]              
([kaldi::discriminati│                 │                      │([kaldi::nnet2][44
ve][441])            │                 │                      │7])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[BackoffDeterministic│[DiscriminativeSu│[LmExampleDeterministi│[OnlineMatrixFeatu│[SpliceMaxComponen
OnDemandFst][448]    │pervision][450]  │cOnDemandFst][452]    │re][454]          │t][456]           
([fst][449])         │([kaldi::discrimi│([fst][453])          │([kaldi][455])    │([kaldi::nnet2][45
                     │native][451])    │                      │                  │7])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BackpointerToken][45│[DiscriminativeSu│[LmState][462]        │[OnlineMatrixInput│[SplitDiscriminati
8]                   │pervisionSplitter│([kaldi][463])        │][464]            │veExampleConfig][4
([kaldi::decoder][459│][460]           │                      │([kaldi][465])    │66]               
])                   │([kaldi::discrimi│                      │                  │([kaldi::nnet2][46
                     │native][461])    │                      │                  │7])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BackpropTruncationCo│[DiscTrainParalle│[MessageLogger::Log][4│[OnlineNaturalGrad│[SplitDiscriminati
mponent][468]        │lClass][470]     │72] ([kaldi][473])    │ient][474]        │veSupervisionOptio
([kaldi::nnet3][469])│([kaldi::nnet2][4│                      │([kaldi::nnet3][47│ns][476]          
                     │71])             │                      │5])               │([kaldi::discrimin
                     │                 │                      │                  │ative][477])      
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BackpropTruncationCo│[DistributeCompon│[MessageLogger::LogAnd│[OnlineNaturalGrad│[SplitEventMap][48
mponentPrecomputedInd│ent][480]        │Throw][482]           │ientSimple][484]  │6] ([kaldi][487]) 
exes][478]           │([kaldi::nnet3][4│([kaldi][483])        │([kaldi::nnet3][48│                  
([kaldi::nnet3][479])│81])             │                      │5])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[basic_filebuf][488] │[DistributeCompon│[LogisticRegression][4│[OnlineNnet2Decodi│[SplitExampleStats
([kaldi][489])       │entPrecomputedInd│92] ([kaldi][493])    │ngConfig][494]    │][496]            
                     │exes][490]       │                      │([kaldi][495])    │([kaldi::nnet2][49
                     │([kaldi::nnet3][4│                      │                  │7])               
                     │91])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[basic_pipebuf][498] │[DoBackpropParall│[LogisticRegressionCon│[OnlineNnet2Decodi│[SplitRadixComplex
([kaldi][499])       │elClass][500]    │fig][502]             │ngThreadedConfig][│Fft][506]         
                     │([kaldi::nnet2][5│([kaldi][503])        │504]              │([kaldi][507])    
                     │01])             │                      │([kaldi][505])    │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BasicHolder][508]   │[ParseOptions::Do│[LogMessageEnvelope][5│[OnlineNnet2Featur│[SplitRadixRealFft
([kaldi][509])       │cInfo][510]      │12] ([kaldi][513])    │ePipeline][514]   │][516]            
                     │([kaldi][511])   │                      │([kaldi][515])    │([kaldi][517])    
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BasicPairVectorHolde│[Dropout][520]   │[LogSoftmaxComponent][│[OnlineNnet2Featur│[SpMatrix][526]   
r][518]              │([kaldi::nnet1][5│522]                  │ePipelineConfig][5│([kaldi][527])    
([kaldi][519])       │21])             │([kaldi::nnet2][523]) │24] ([kaldi][525])│                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BasicVectorHolder][5│[DropoutComponent│[LogSoftmaxComponent][│[OnlineNnet2Featur│[StandardInputImpl
28] ([kaldi][529])   │][530]           │532]                  │ePipelineInfo][534│][536]            
                     │([kaldi::nnet2][5│([kaldi::nnet3][533]) │] ([kaldi][535])  │([kaldi][537])    
                     │31])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BasicVectorVectorHol│[DropoutComponent│[LossItf][542]        │[OnlinePaSource][5│[StandardOutputImp
der][538]            │][540]           │([kaldi::nnet1][543]) │44] ([kaldi][545])│l][546]           
([kaldi][539])       │([kaldi::nnet3][5│                      │                  │([kaldi][547])    
                     │41])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BasisFmllrAccus][548│[DropoutMaskCompo│[LossOptions][552]    │[OnlinePitchFeatur│[PitchFrameInfo::S
] ([kaldi][549])     │nent][550]       │([kaldi::nnet1][553]) │e][554]           │tateInfo][556]    
                     │([kaldi::nnet3][5│                      │([kaldi][555])    │([kaldi][557])    
                     │51])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BasisFmllrEstimate][│[DummyOptions][56│[LstmNonlinearityCompo│[OnlinePitchFeatur│[StateIterator<   
558] ([kaldi][559])  │0] ([kaldi][561])│nent][562]            │eImpl][564]       │TrivialFactorWeigh
                     │                 │([kaldi::nnet3][563]) │([kaldi][565])    │tFst< A, F >      
                     │                 │                      │                  │>][566]           
                     │                 │                      │                  │([fst][567])      
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BasisFmllrOptions][5│e                │[LstmProjected][570]  │[OnlinePreconditio│[StatisticsExtract
68] ([kaldi][569])   │                 │([kaldi::nnet1][571]) │ner][572]         │ionComponent][574]
                     │                 │                      │([kaldi::nnet2][57│([kaldi::nnet3][57
                     │                 │                      │3])               │5])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[BatchedXvectorComput│m                │[OnlinePreconditionerS│[StatisticsExtract
er][576]             │                 │imple][578]           │ionComponentPrecom
([kaldi::nnet3][577])│                 │([kaldi::nnet2][579]) │putedIndexes][580]
                     │                 │                      │([kaldi::nnet3][58
                     │                 │                      │1])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────
[BatchedXvectorComput│[EbwAmSgmm2Option│[OnlineProcessPitch][5│[StatisticsPooling
erOptions][582]      │s][584]          │86] ([kaldi][587])    │Component][588]   
([kaldi::nnet3][583])│([kaldi][585])   │                      │([kaldi::nnet3][58
                     │                 │                      │9])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[BatchNormComponent][│[EbwAmSgmm2Update│[MapDiagGmmOptions][59│[OnlineSilenceWeig│[StatisticsPooling
590]                 │r][592]          │4] ([kaldi][595])     │hting][596]       │ComponentPrecomput
([kaldi::nnet3][591])│([kaldi][593])   │                      │([kaldi][597])    │edIndexes][598]   
                     │                 │                      │                  │([kaldi::nnet3][59
                     │                 │                      │                  │9])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[LatticeFasterOnlineD│[EbwAmSgmmUpdater│[MapInputSymbolsMapper│[OnlineSilenceWeig│[NnetStats::StatsE
ecoderTpl::BestPathIt│][602]           │][603] ([fst][604])   │htingConfig][605] │lement][607]      
erator][600]         │                 │                      │([kaldi][606])    │([kaldi::nnet2][60
([kaldi][601])       │                 │                      │                  │8])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[LatticeIncrementalOn│[EbwOptions][611]│[MapTransitionUpdateCo│[OnlineSpeexDecode│[ExampleMergingSta
lineDecoderTpl::BestP│([kaldi][612])   │nfig][613]            │r][615]           │ts::StatsForExampl
athIterator][609]    │                 │([kaldi][614])        │([kaldi][616])    │eSize][617]       
([kaldi][610])       │                 │                      │                  │([kaldi::nnet3][61
                     │                 │                      │                  │8])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BiglmFasterDecoder][│[EbwUpdatePhoneVe│[Matrix][623]         │[OnlineSpeexEncode│[StdToken][627]   
619] ([kaldi][620])  │ctorsClass][621] │([kaldi][624])        │r][625]           │([kaldi::decoder][
                     │([kaldi][622])   │                      │([kaldi][626])    │628])             
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BiglmFasterDecoderOp│[EbwWeightOptions│[MatrixAccesses][633] │[OnlineSpliceFrame│[StdToLatticeMappe
tions][629]          │][631]           │([kaldi::nnet3][634]) │s][635]           │r][637]           
([kaldi][630])       │([kaldi][632])   │                      │([kaldi][636])    │([fst][638])      
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BinarySumDescriptor]│[EigenvalueDecomp│[MatrixBase][643]     │[OnlineSpliceOptio│[StdVectorRandomiz
[639]                │osition][641]    │([kaldi][644])        │ns][645]          │er][647]          
([kaldi::nnet3][640])│([kaldi][642])   │                      │([kaldi][646])    │([kaldi::nnet1][64
                     │                 │                      │                  │8])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BlockAffineComponent│[HashList::Elem][│[MatrixBuffer][653]   │[OnlineTcpVectorSo│[Compiler::StepInf
][649]               │651]             │([kaldi::nnet1][654]) │urce][655]        │o][657]           
([kaldi::nnet2][650])│([kaldi][652])   │                      │([kaldi][656])    │([kaldi::nnet3][65
                     │                 │                      │                  │8])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BlockAffineComponent│[LatticeDetermini│[MatrixBufferOptions][│[OnlineTimer][665]│[StringHasher][667
][659]               │zer::Element][661│663]                  │([kaldi][666])    │] ([kaldi][668])  
([kaldi::nnet3][660])│] ([fst][662])   │([kaldi::nnet1][664]) │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BlockAffineComponent│[DeterminizerStar│[MemoryCompressionOpti│[OnlineTimingStats│[StringRepository]
Preconditioned][669] │::Element][671]  │mizer::MatrixCompressI│][675]            │[677] ([fst][678])
([kaldi::nnet2][670])│([fst][672])     │nfo][673]             │([kaldi][676])    │                  
                     │                 │([kaldi::nnet3][674]) │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BlockFactorizedTdnnC│[TrivialFactorWei│[NnetComputation::Matr│[OnlineTransform][│[SubMatrix][687]  
omponent][679]       │ghtFstImpl::Eleme│ixDebugInfo][683]     │685]              │([kaldi][688])    
([kaldi::nnet3][680])│nt][681]         │([kaldi::nnet3][684]) │([kaldi][686])    │                  
                     │([fst::internal][│                      │                  │                  
                     │682])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuBlockMatrix::Block│[LatticeDetermini│[MatrixDim_][693]     │[OnlineUdpInput][6│[ComputationRenumb
MatrixData][689]     │zerPruned::Elemen│                      │94] ([kaldi][695])│erer::SubMatrixHas
([kaldi][690])       │t][691]          │                      │                  │her][696]         
                     │([fst][692])     │                      │                  │([kaldi::nnet3][69
                     │                 │                      │                  │7])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BlockSoftmax][698]  │[TrivialFactorWei│[MatrixElement][702]  │[OnlineVectorSourc│[NnetComputation::
([kaldi::nnet1][699])│ghtFstImpl::Eleme│                      │e][703]           │SubMatrixInfo][705
                     │ntEqual][700]    │                      │([kaldi][704])    │]                 
                     │([fst::internal][│                      │                  │([kaldi::nnet3][70
                     │701])            │                      │                  │6])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BlstmProjected][707]│[TrivialFactorWei│[MatrixExtender][711] │[OptimizableInterf│[DeterminizerStar:
([kaldi::nnet1][708])│ghtFstImpl::Eleme│([kaldi::nnet3][712]) │ace][713]         │:SubsetEqual][715]
                     │ntKey][709]      │                      │([kaldi][714])    │([fst][716])      
                     │([fst::internal][│                      │                  │                  
                     │710])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[BottomUpClusterer][7│[ElementwiseProdu│[NnetComputation::Matr│[OptimizeLbfgs][72│[LatticeDeterminiz
17] ([kaldi][718])   │ctComponent][719]│ixInfo][721]          │3] ([kaldi][724]) │erPruned::SubsetEq
                     │([kaldi::nnet3][7│([kaldi::nnet3][722]) │                  │ual][725]         
                     │20])             │                      │                  │([fst][726])      
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
c                    │[LatticeStringRep│[DerivativeTimeLimiter│[OptionalSumDescri│[LatticeDeterminiz
                     │ository::Entry][7│::MatrixPruneInfo][729│ptor][731]        │er::SubsetEqual][7
                     │27] ([fst][728]) │]                     │([kaldi::nnet3][73│33] ([fst][734])  
                     │                 │([kaldi::nnet3][730]) │2])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[LatticeStringReposit│[MatrixRandomizer│[SimpleOptions::Option│[LatticeDeterminiz
ory::EntryEqual][735]│][737]           │Info][739]            │er::SubsetEqualSta
([fst][736])         │([kaldi::nnet1][7│([kaldi][740])        │tes][741]         
                     │38])             │                      │([fst][742])      
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[CacheArcIterator][74│[LatticeStringRep│[MaxChangeStats][746] │[OptionsItf][748] │[DeterminizerStar:
3]                   │ository::EntryKey│([kaldi::nnet3][747]) │([kaldi][749])    │:SubsetEqualStates
                     │][744]           │                      │                  │][750]            
                     │([fst][745])     │                      │                  │([fst][751])      
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CacheDeterministicOn│[DeterminizerStar│[MaxoutComponent][756]│[OtherReal][758]  │[LatticeDeterminiz
DemandFst][752]      │::EpsilonClosure]│([kaldi::nnet2][757]) │([kaldi][759])    │erPruned::SubsetEq
([fst][753])         │[754]            │                      │                  │ualStates][760]   
                     │([fst][755])     │                      │                  │([fst][761])      
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CacheImpl][762]     │[DeterminizerStar│[MaxpoolingComponent][│[OtherReal< double│[DeterminizerStar:
                     │::EpsilonClosure:│765]                  │>][767]           │:SubsetKey][769]  
                     │:EpsilonClosureIn│([kaldi::nnet2][766]) │([kaldi][768])    │([fst][770])      
                     │fo][763]         │                      │                  │                  
                     │([fst][764])     │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CacheOptions][771]  │[CompactLatticeMi│[MaxpoolingComponent][│[OtherReal< float │[LatticeDeterminiz
                     │nimizer::Equivale│774]                  │>][776]           │er::SubsetKey][778
                     │nceSorter][772]  │([kaldi::nnet3][775]) │([kaldi][777])    │] ([fst][779])    
                     │([fst][773])     │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CacheStateIterator][│[error_stats][781│[MaxPoolingComponent][│[Output][785]     │[LatticeDeterminiz
780]                 │] ([kaldi][782]) │783]                  │([kaldi][786])    │erPruned::SubsetKe
                     │                 │([kaldi::nnet1][784]) │                  │y][787]           
                     │                 │                      │                  │([fst][788])      
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CachingOptimizingCom│[EventMap][791]  │[MelBanks][793]       │[OutputImplBase][7│[Sgmm2LikelihoodCa
piler][789]          │([kaldi][792])   │([kaldi][794])        │95] ([kaldi][796])│che::SubstateCache
([kaldi::nnet3][790])│                 │                      │                  │Element][797]     
                     │                 │                      │                  │([kaldi][798])    
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CachingOptimizingCom│[EventMapVectorEq│[MelBanksOptions][803]│[LatticeDeterminiz│[SubVector][807]  
pilerOptions][799]   │ual][801]        │([kaldi][804])        │erPruned::OutputSt│([kaldi][808])    
([kaldi::nnet3][800])│([kaldi][802])   │                      │ate][805]         │                  
                     │                 │                      │([fst][806])      │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ChainExampleMerger][│[EventMapVectorHa│[RestrictedAttentionCo│p                 │[SumBlockComponent
809]                 │sh][811]         │mponent::Memo][813]   │                  │][815]            
([kaldi::nnet3][810])│([kaldi][812])   │([kaldi::nnet3][814]) │                  │([kaldi::nnet3][81
                     │                 │                      │                  │6])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[ChainObjectiveInfo][│[ExampleClass][81│[BatchNormComponent::M│[SumDescriptor][82
817]                 │9] ([kaldi][820])│emo][821]             │3]                
([kaldi::nnet3][818])│                 │([kaldi::nnet3][822]) │([kaldi::nnet3][82
                     │                 │                      │4])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[CheckComputationOpti│[ExampleFeatureCo│[MemoryCompressionOpti│[PackedMatrix][831│[SumGroupComponent
ons][825]            │mputer][827]     │mizer][829]           │] ([kaldi][832])  │][833]            
([kaldi::nnet3][826])│([kaldi][828])   │([kaldi::nnet3][830]) │                  │([kaldi::nnet2][83
                     │                 │                      │                  │4])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[LmState::ChildrenVec│[ExampleFeatureCo│[MessageLogger][839]  │[LatticeDeterminiz│[SumGroupComponent
torLessThan][835]    │mputerOptions][83│([kaldi][840])        │er::PairComparator│][843]            
([kaldi][836])       │7] ([kaldi][838])│                      │][841]            │([kaldi::nnet3][84
                     │                 │                      │([fst][842])      │4])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[LmState::ChildType][│[ExampleGeneratio│[MfccComputer][849]   │[DeterminizerStar:│[SvdApplier][853] 
845] ([kaldi][846])  │nConfig][847]    │([kaldi][850])        │:PairComparator][8│([kaldi::nnet3][85
                     │([kaldi::nnet3][8│                      │51] ([fst][852])  │4])               
                     │48])             │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ChunkInfo][855]     │[ExampleMerger][8│[MfccOptions][859]    │[LatticeDeterminiz│[SwitchingForwardi
([kaldi::nnet2][856])│57]              │([kaldi][860])        │erPruned::PairComp│ngDescriptor][863]
                     │([kaldi::nnet3][8│                      │arator][861]      │([kaldi::nnet3][86
                     │58])             │                      │([fst][862])      │4])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ChunkInfo][865]     │[ExampleMergingCo│[SimpleMeanTransform::│[RandomAccessTable│[synapse][873]    
([kaldi::nnet3][866])│nfig][867]       │MinibatchInfo][869]   │ReaderSortedArchiv│([rnnlm][874])    
                     │([kaldi::nnet3][8│([kaldi::differentiabl│eImpl::PairCompare│                  
                     │68])             │e_transform][870])    │][871]            │                  
                     │                 │                      │([kaldi][872])    │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ChunkTimeInfo][875] │[ExampleMergingSt│[FmllrTransform::Minib│[PairHasher][881] │t                 
([kaldi::nnet3][876])│ats][877]        │atchInfo][879]        │([kaldi][882])    │                  
                     │([kaldi::nnet3][8│([kaldi::differentiabl│                  │                  
                     │78])             │e_transform][880])    │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[CindexHasher][883]  │[ExamplesReposito│[MinibatchInfoItf][887│[PairIsEqualCompar
([kaldi::nnet3][884])│ry][885]         │]                     │ator][889]        
                     │([kaldi::nnet2][8│([kaldi::differentiabl│([kaldi::nnet3][89
                     │86])             │e_transform][888])    │0])               
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[ComputationGraphBuil│[GrammarFst::Expa│[NnetBatchComputer::Mi│[ParallelComponent│[TableComposeCache
der::CindexInfo][891]│ndedState][893]  │nibatchSizeInfo][895] │][897]            │][899]            
([kaldi::nnet3][892])│([fst][894])     │([kaldi::nnet3][896]) │([kaldi::nnet1][89│([fst][900])      
                     │                 │                      │8])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CindexSet][901]     │f                │[MinimumBayesRisk][903│[ParametricRelu][9│[TableComposeOptio
([kaldi::nnet3][902])│                 │] ([kaldi][904])      │05]               │ns][907]          
                     │                 │                      │([kaldi::nnet1][90│([fst][908])      
                     │                 │                      │6])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[CindexVectorHasher][│[MinimumBayesRisk│[ParseOptions][913]   │[TableEventMap][91
909]                 │Options][911]    │([kaldi][914])        │5] ([kaldi][916]) 
([kaldi::nnet3][910])│([kaldi][912])   │                      │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[PldaStats::ClassInfo│[FasterDecoder][9│[MiscComputationInfo][│[Sgmm2LikelihoodCa│[TableMatcher][925
][917] ([kaldi][918])│19]              │921]                  │che::PdfCacheEleme│] ([fst][926])    
                     │([kaldi][920])   │([kaldi::nnet3][922]) │nt][923]          │                  
                     │                 │                      │([kaldi][924])    │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ClatRescoreTuple][92│[FasterDecoderOpt│[MleAmSgmm2Accs][931] │[PdfPrior][933]   │[TableMatcherImpl]
7] ([kaldi][928])    │ions][929]       │([kaldi][932])        │([kaldi::nnet1][93│[935] ([fst][936])
                     │([kaldi][930])   │                      │4])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ClipGradientComponen│[FastNnetCombiner│[MleAmSgmm2Options][94│[PdfPriorOptions][│[TableMatcherOptio
t][937]              │][939]           │1] ([kaldi][942])     │943]              │ns][945]          
([kaldi::nnet3][938])│([kaldi::nnet2][9│                      │([kaldi::nnet1][94│([fst][946])      
                     │40])             │                      │4])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[Clusterable][947]   │[FbankComputer][9│[MleAmSgmm2Updater][95│[CompressedMatrix:│[TableWriter][955]
([kaldi][948])       │49]              │1] ([kaldi][952])     │:PerColHeader][953│([kaldi][956])    
                     │([kaldi][950])   │                      │] ([kaldi][954])  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ClusterKMeansOptions│[FbankOptions][95│[MleAmSgmmUpdater][961│[PerDimObjectiveIn│[TableWriterArchiv
][957] ([kaldi][958])│9] ([kaldi][960])│]                     │fo][962]          │eImpl][964]       
                     │                 │                      │([kaldi::nnet3][96│([kaldi][965])    
                     │                 │                      │3])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CollapseModelConfig]│[FeatureTransform│[MleDiagGmmOptions][97│[PerElementOffsetC│[TableWriterBothIm
[966]                │Estimate][968]   │0] ([kaldi][971])     │omponent][972]    │pl][974]          
([kaldi::nnet3][967])│([kaldi][969])   │                      │([kaldi::nnet3][97│([kaldi][975])    
                     │                 │                      │3])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[NnetComputation::Com│[FeatureTransform│[MleFullGmmOptions][98│[PerElementScaleCo│[TableWriterImplBa
mand][976]           │EstimateMulti][97│0] ([kaldi][981])     │mponent][982]     │se][984]          
([kaldi::nnet3][977])│8] ([kaldi][979])│                      │([kaldi::nnet3][98│([kaldi][985])    
                     │                 │                      │3])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CommandAttributes][9│[FeatureTransform│[MleSgmm2SpeakerAccs][│[PermuteComponent]│[TableWriterScript
86]                  │EstimateOptions][│990] ([kaldi][991])   │[992]             │Impl][994]        
([kaldi::nnet3][987])│988]             │                      │([kaldi::nnet2][99│([kaldi][995])    
                     │([kaldi][989])   │                      │3])               │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[NnetComputer::Comman│[FeatureWindowFun│[MleTransitionUpdateCo│[PermuteComponent]│[Tanh][1004]      
dDebugInfo][996]     │ction][998]      │nfig][1000]           │[1002]            │([kaldi::nnet1][10
([kaldi::nnet3][997])│([kaldi][999])   │([kaldi][1001])       │([kaldi::nnet3][10│05])              
                     │                 │                      │03])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CommandPairComparato│[FileInputImpl][1│[MlltAccs][1010]      │[PhoneAlignLattice│[TanhComponent][10
r][1006]             │008]             │([kaldi][1011])       │Options][1012]    │14]               
([kaldi::nnet3][1007]│([kaldi][1009])  │                      │([kaldi][1013])   │([kaldi::nnet3][10
)                    │                 │                      │                  │15])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompactLatticeHolder│[FileOutputImpl][│[ModelCollapser][1020]│[PipeInputImpl][10│[TanhComponent][10
][1016]              │1018]            │([kaldi::nnet3][1021])│22]               │24]               
([kaldi][1017])      │([kaldi][1019])  │                      │([kaldi][1023])   │([kaldi::nnet2][10
                     │                 │                      │                  │25])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompactLatticeMinimi│[FisherComputatio│[ModelUpdateConsolidat│[PipeOutputImpl][1│[TarjanNode][1034]
zer][1026]           │nClass][1028]    │or][1030]             │032]              │([kaldi::nnet3][10
([fst][1027])        │([kaldi::nnet2][1│([kaldi::nnet3][1031])│([kaldi][1033])   │35])              
                     │029])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompactLatticePusher│[FixedAffineCompo│[SvdApplier::ModifiedC│[PitchExtractionOp│[PruneSpecialClass
][1036] ([fst][1037])│nent][1038]      │omponentInfo][1040]   │tions][1042]      │::Task][1044]     
                     │([kaldi::nnet2][1│([kaldi::nnet3][1041])│([kaldi][1043])   │([fst][1045])     
                     │039])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompactLatticeToKwsP│[FixedAffineCompo│[Mse][1050]           │[PitchFrameInfo][1│[LatticeDeterminiz
roductFstMapper][1046│nent][1048]      │([kaldi::nnet1][1051])│052]              │erPruned::Task][10
] ([kaldi][1047])    │([kaldi::nnet3][1│                      │([kaldi][1053])   │54] ([fst][1055]) 
                     │049])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompactLatticeWeight│[FixedBiasCompone│[MultiBasisComponent][│[PitchInterpolator│[LatticeDeterminiz
CommonDivisorTpl][105│nt][1058]        │1060]                 │][1062]           │erPruned::TaskComp
6] ([fst][1057])     │([kaldi::nnet2][1│([kaldi::nnet1][1061])│([kaldi][1063])   │are][1064]        
                     │059])            │                      │                  │([fst][1065])     
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompactLatticeWeight│[FixedBiasCompone│[RowOpsSplitter::Multi│[PitchInterpolator│[TaskSequencer][10
Tpl][1066]           │nt][1068]        │IndexSplitInfo][1070] │Options][1072]    │74]               
([fst][1067])        │([kaldi::nnet3][1│([kaldi::nnet3][1071])│([kaldi][1073])   │([kaldi][1075])   
                     │069])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompareFirst][1076] │[FixedLinearCompo│[MultistreamComponent]│[PitchInterpolator│[TaskSequencerConf
([kaldi::sparse_vecto│nent][1078]      │[1080]                │Stats][1082]      │ig][1084]         
r_utils][1077])      │([kaldi::nnet2][1│([kaldi::nnet1][1081])│([kaldi][1083])   │([kaldi][1085])   
                     │079])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompareFirstMemberOf│[FixedScaleCompon│[MultiTaskLoss][1090] │[Plda][1092]      │[TcpServer][1094] 
Pair][1086]          │ent][1088]       │([kaldi::nnet1][1091])│([kaldi][1093])   │([kaldi][1095])   
([kaldi][1087])      │([kaldi::nnet2][1│                      │                  │                  
                     │089])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComparePair][1096]  │[FixedScaleCompon│[MultiThreadable][1100│[PldaConfig][1102]│[TdnnComponent][11
([kaldi::nnet3][1097]│ent][1098]       │] ([kaldi][1101])     │([kaldi][1103])   │04]               
)                    │([kaldi::nnet3][1│                      │                  │([kaldi::nnet3][11
                     │099])            │                      │                  │05])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComparePosteriorByPd│[FloatWeightTpl][│[MultiThreader][1109] │[PldaEstimationCon│[LatticeDeterminiz
fs][1106]            │1108]            │([kaldi][1110])       │fig][1111]        │er::TempArc][1113]
([kaldi][1107])      │                 │                      │([kaldi][1112])   │([fst][1114])     
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompareReverseSecond│[FmllrDiagGmmAccs│[MyTaskClass][1119]   │[PldaEstimator][11│[LatticeDeterminiz
][1115]              │][1117]          │([kaldi][1120])       │21]               │erPruned::TempArc]
([kaldi][1116])      │([kaldi][1118])  │                      │([kaldi][1122])   │[1123]            
                     │                 │                      │                  │([fst][1124])     
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompartmentalizedBot│[FmllrOptions][11│[MyThreadClass][1129] │[PldaStats][1131] │[DeterminizerStar:
tomUpClusterer][1125]│27]              │([kaldi][1130])       │([kaldi][1132])   │:TempArc][1133]   
([kaldi][1126])      │([kaldi][1128])  │                      │                  │([fst][1134])     
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompBotClustElem][11│[FmllrRawAccs][11│n                     │[PldaUnsupervisedA│[TestFunction][114
35] ([kaldi][1136])  │37]              │                      │daptor][1139]     │1]                
                     │([kaldi][1138])  │                      │([kaldi][1140])   │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[Compiler][1142]     │[FmllrRawOptions]│[PldaUnsupervisedAdapt│[TestFunctor][1148
([kaldi::nnet3][1143]│[1144]           │orConfig][1146]       │] ([fst][1149])   
)                    │([kaldi][1145])  │([kaldi][1147])       │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[CompilerOptions][115│[FmllrSgmm2Accs][│[NaturalGradientAffine│[PlpComputer][1156│[ThreadSynchronize
0]                   │1152]            │Component][1154]      │] ([kaldi][1157]) │r][1158]          
([kaldi::nnet3][1151]│([kaldi][1153])  │([kaldi::nnet3][1155])│                  │([kaldi][1159])   
)                    │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[Component][1160]    │[FmllrTransform][│[NaturalGradientPerEle│[PlpOptions][1166]│[ThrSweepStats][11
([kaldi::nnet2][1161]│1162]            │mentScaleComponent][11│([kaldi][1167])   │68]               
)                    │([kaldi::differen│64]                   │                  │([kaldi::kws_inter
                     │tiable_transform]│([kaldi::nnet3][1165])│                  │nal][1169])       
                     │[1163])          │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[Component][1170]    │[Fmpe][1172]     │[NaturalGradientRepeat│[PnormComponent][1│[TidToTstateMapper
([kaldi::nnet3][1171]│([kaldi][1173])  │edAffineComponent][117│176]              │][1178]           
)                    │                 │4]                    │([kaldi::nnet2][11│([kaldi][1179])   
                     │                 │([kaldi::nnet3][1175])│77])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[Component][1180]    │[FmpeOptions][118│[NaturalLess<         │[PnormComponent][1│[TimeHeightConvolu
([kaldi::nnet1][1181]│2]               │CompactLatticeWeightTp│186]              │tionComponent][118
)                    │([kaldi][1183])  │l< LatticeWeightTpl<  │([kaldi::nnet3][11│8]                
                     │                 │double >, int32 >     │87])              │([kaldi::nnet3][11
                     │                 │>][1184] ([fst][1185])│                  │89])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComponentPrecomputed│[FmpeStats][1192]│[NaturalLess<         │[RefineClusterer::│[Timer][1198]     
Indexes][1190]       │([kaldi][1193])  │CompactLatticeWeightTp│point_info][1196] │([kaldi][1199])   
([kaldi::nnet3][1191]│                 │l< LatticeWeightTpl<  │([kaldi][1197])   │                  
)                    │                 │float >, int32 >      │                  │                  
                     │                 │>][1194] ([fst][1195])│                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComposeDeterministic│[FmpeUpdateOption│[NaturalLess<         │[ComputationRenumb│[SimpleDecoder::To
OnDemandFst][1200]   │s][1202]         │CompactLatticeWeightTp│erer::PointerCompa│ken][1208]        
([fst][1201])        │([kaldi][1203])  │l< LatticeWeightTpl<  │re][1206]         │([kaldi][1209])   
                     │                 │FloatType >, IntType >│([kaldi::nnet3][12│                  
                     │                 │>][1204] ([fst][1205])│07])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[PrunedCompactLattice│[ForwardingDescri│[NaturalLess<         │[PosteriorHolder][│[FasterDecoder::To
Composer::ComposedSta│ptor][1212]      │LatticeWeightTpl<     │1216]             │ken][1218]        
teInfo][1210]        │([kaldi::nnet3][1│double > >][1214]     │([kaldi][1217])   │([kaldi][1219])   
([kaldi][1211])      │213])            │([fst][1215])         │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComposeLatticePruned│[LatticeBiglmFast│[NaturalLess<         │[PowerComponent][1│[LatticeBiglmFaste
Options][1220]       │erDecoder::Forwar│LatticeWeightTpl<     │226]              │rDecoder::Token][1
([kaldi][1221])      │dLink][1222]     │float > >][1224]      │([kaldi::nnet2][12│228]              
                     │([kaldi][1223])  │([fst][1225])         │27])              │([kaldi][1229])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompositeComponent][│[ForwardLink][123│[NaturalLess<         │[RestrictedAttenti│[LatticeSimpleDeco
1230]                │2]               │LatticeWeightTpl<     │onComponent::Preco│der::Token][1238] 
([kaldi::nnet3][1231]│([kaldi::decoder]│FloatType > >][1234]  │mputedIndexes][123│([kaldi][1239])   
)                    │[1233])          │([fst][1235])         │6]                │                  
                     │                 │                      │([kaldi::nnet3][12│                  
                     │                 │                      │37])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompressedAffineXfor│[LatticeSimpleDec│[NccfInfo][1244]      │[TimeHeightConvolu│[BiglmFasterDecode
mStats][1240]        │oder::ForwardLink│([kaldi][1245])       │tionComponent::Pre│r::Token][1248]   
([kaldi][1241])      │][1242]          │                      │computedIndexes][1│([kaldi][1249])   
                     │([kaldi][1243])  │                      │246]              │                  
                     │                 │                      │([kaldi::nnet3][12│                  
                     │                 │                      │47])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CompressedMatrix][12│[FrameExtractionO│[NetworkNode][1254]   │[TdnnComponent::Pr│[TokenHolder][1258
50] ([kaldi][1251])  │ptions][1252]    │([kaldi::nnet3][1255])│ecomputedIndexes][│] ([kaldi][1259]) 
                     │([kaldi][1253])  │                      │1256]             │                  
                     │                 │                      │([kaldi::nnet3][12│                  
                     │                 │                      │57])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationAnalysis]│[DiscriminativeEx│[neuron][1264]        │[NnetComputation::│[LatticeIncrementa
[1260]               │ampleSplitter::Fr│([rnnlm][1265])       │PrecomputedIndexes│lDecoderTpl::Token
([kaldi::nnet3][1261]│ameInfo][1262]   │                      │Info][1266]       │List][1268]       
)                    │([kaldi::nnet2][1│                      │([kaldi::nnet3][12│([kaldi][1269])   
                     │263])            │                      │67])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationCache][12│[OnlineSilenceWei│[NGram][1274]         │[ProcessPitchOptio│[LatticeSimpleDeco
70]                  │ghting::FrameInfo│([kaldi][1275])       │ns][1276]         │der::TokenList][12
([kaldi::nnet3][1271]│][1272]          │                      │([kaldi][1277])   │78]               
)                    │([kaldi][1273])  │                      │                  │([kaldi][1279])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationChecker][│[FramePoolingComp│[Nnet][1284]          │[Profiler][1286]  │[LatticeBiglmFaste
1280]                │onent][1282]     │([kaldi::nnet2][1285])│([kaldi][1287])   │rDecoder::TokenLis
([kaldi::nnet3][1281]│([kaldi::nnet1][1│                      │                  │t][1288]          
)                    │283])            │                      │                  │([kaldi][1289])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationExpander]│[GrammarFst::FstI│[Nnet][1294]          │[ProfileStats][129│[LatticeFasterDeco
[1290]               │nstance][1292]   │([kaldi::nnet3][1295])│6] ([kaldi][1297])│derTpl::TokenList]
([kaldi::nnet3][1291]│([fst][1293])    │                      │                  │[1298]            
)                    │                 │                      │                  │([kaldi][1299])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationGraph][13│[FullGmm][1302]  │[Nnet][1304]          │[ProfileStats::Pro│[TokenVectorHolder
00]                  │([kaldi][1303])  │([kaldi::nnet1][1305])│fileStatsEntry][13│][1308]           
([kaldi::nnet3][1301]│                 │                      │06]               │([kaldi][1309])   
)                    │                 │                      │([kaldi][1307])   │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationGraphBuil│[FullGmmNormal][1│[NnetBatchComputer][13│[PrunedCompactLatt│[TpMatrix][1318]  
der][1310]           │312]             │14]                   │iceComposer][1316]│([kaldi][1319])   
([kaldi::nnet3][1311]│([kaldi][1313])  │([kaldi::nnet3][1315])│([kaldi][1317])   │                  
)                    │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[NnetBatchComputer::C│g                │[NnetBatchComputerOpti│[PruneSpecialClass│[TrainingGraphComp
omputationGroupInfo][│                 │ons][1322]            │][1324]           │iler][1326]       
1320]                │                 │([kaldi::nnet3][1323])│([fst][1325])     │([kaldi][1327])   
([kaldi::nnet3][1321]│                 │                      │                  │                  
)                    │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[NnetBatchComputer::C│[NnetBatchDecoder│[PushSpecialClass][133│[TrainingGraphComp
omputationGroupKey][1│][1330]          │2] ([fst][1333])      │ilerOptions][1334]
328]                 │([kaldi::nnet3][1│                      │([kaldi][1335])   
([kaldi::nnet3][1329]│331])            │                      │                  
)                    │                 │                      │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[NnetBatchComputer::C│[MinimumBayesRisk│[NnetBatchInference][1│q                 │[TransitionModel][
omputationGroupKeyHas│::GammaCompare][1│340]                  │                  │1342]             
her][1336]           │338]             │([kaldi::nnet3][1341])│                  │([kaldi][1343])   
([kaldi::nnet3][1337]│([kaldi][1339])  │                      │                  │                  
)                    │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[ComputationLoopedOpt│[GaussClusterable│[NnetChainComputeProb]│[TreeClusterer][13
imizer][1344]        │][1346]          │[1348]                │50]               
([kaldi::nnet3][1345]│([kaldi][1347])  │([kaldi::nnet3][1349])│([kaldi][1351])   
)                    │                 │                      │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[ComputationRenumbere│[GaussInfo][1354]│[NnetChainExample][135│[Questions][1358] │[TreeClusterOption
r][1352]             │([kaldi][1355])  │6]                    │([kaldi][1359])   │s][1360]          
([kaldi::nnet3][1353]│                 │([kaldi::nnet3][1357])│                  │([kaldi][1361])   
)                    │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationRequest][│[GaussPostHolder]│[NnetChainExampleStruc│[QuestionsForKey][│[TreeRenderer][137
1362]                │[1364]           │tureCompare][1366]    │1368]             │0] ([kaldi][1371])
([kaldi::nnet3][1363]│([kaldi][1365])  │([kaldi::nnet3][1367])│([kaldi][1369])   │                  
)                    │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationRequestHa│[GeneralDescripto│[NnetChainExampleStruc│r                 │[TrivialFactorWeig
sher][1372]          │r][1374]         │tureHasher][1376]     │                  │htFst][1378]      
([kaldi::nnet3][1373]│([kaldi::nnet3][1│([kaldi::nnet3][1377])│                  │([fst][1379])     
)                    │375])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[ComputationRequestPt│[GeneralDropoutCo│[NnetChainSupervision]│[TrivialFactorWeig
rEqual][1380]        │mponent][1382]   │[1384]                │htFstImpl][1386]  
([kaldi::nnet3][1381]│([kaldi::nnet3][1│([kaldi::nnet3][1385])│([fst::internal][1
)                    │383])            │                      │387])             
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[LatticePhoneAligner:│[GeneralDropoutCo│[NnetChainTrainer][139│[RandFstOptions][1│[TrivialFactorWeig
:ComputationState][13│mponentPrecompute│2]                    │394] ([fst][1395])│htOptions][1396]  
88] ([kaldi][1389])  │dIndexes][1390]  │([kaldi::nnet3][1393])│                  │([fst][1397])     
                     │([kaldi::nnet3][1│                      │                  │                  
                     │391])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[LatticeLexiconWordAl│[GeneralMatrix][1│[NnetChainTrainingOpti│[RandomAccessTable│[LatticeLexiconWor
igner::ComputationSta│400]             │ons][1402]            │Reader][1404]     │dAligner::Tuple][1
te][1398]            │([kaldi][1401])  │([kaldi::nnet3][1403])│([kaldi][1405])   │406]              
([kaldi][1399])      │                 │                      │                  │([kaldi][1407])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[LatticeWordAligner::│[GenericHolder][1│[NnetCombineAconfig][1│[RandomAccessTable│[LatticePhoneAlign
ComputationState][140│410]             │412]                  │ReaderArchiveImplB│er::Tuple][1416]  
8] ([kaldi][1409])   │([kaldi][1411])  │([kaldi::nnet2][1413])│ase][1414]        │([kaldi][1417])   
                     │                 │                      │([kaldi][1415])   │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationStepsComp│[CompressedMatrix│[NnetCombineConfig][14│[RandomAccessTable│[LatticeWordAligne
uter][1418]          │::GlobalHeader][1│22]                   │ReaderDSortedArchi│r::Tuple][1426]   
([kaldi::nnet3][1419]│420]             │([kaldi::nnet2][1423])│veImpl][1424]     │([kaldi][1427])   
)                    │([kaldi][1421])  │                      │([kaldi][1425])   │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputationVariables│[GrammarFst][1430│[NnetCombineFastConfig│[RandomAccessTable│[TransitionModel::
][1428]              │] ([fst][1431])  │][1432]               │ReaderImplBase][14│Tuple][1436]      
([kaldi::nnet3][1429]│                 │([kaldi::nnet2][1433])│34]               │([kaldi][1437])   
)                    │                 │                      │([kaldi][1435])   │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ComputeNormalizersCl│[GrammarFstArc][1│[NnetComputation][1442│[RandomAccessTable│[LatticeWordAligne
ass][1438]           │440]             │]                     │ReaderMapped][1444│r::TupleEqual][144
([kaldi][1439])      │([fst][1441])    │([kaldi::nnet3][1443])│] ([kaldi][1445]) │6] ([kaldi][1447])
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConfigLine][1448]   │[GrammarFstPrepar│[NnetComputationPrintI│[RandomAccessTable│[LatticeLexiconWor
([kaldi][1449])      │er][1450]        │nserter][1452]        │ReaderScriptImpl][│dAligner::TupleEqu
                     │([fst][1451])    │([kaldi::nnet3][1453])│1454]             │al][1456]         
                     │                 │                      │([kaldi][1455])   │([kaldi][1457])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConstantComponent][1│h                │[NnetComputeOptions][1│[RandomAccessTable│[LatticePhoneAlign
458]                 │                 │460]                  │ReaderSortedArchiv│er::TupleEqual][14
([kaldi::nnet3][1459]│                 │([kaldi::nnet3][1461])│eImpl][1462]      │64]               
)                    │                 │                      │([kaldi][1463])   │([kaldi][1465])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[ConstantEventMap][14│[NnetComputeProb]│[RandomAccessTableRead│[LatticeLexiconWor
66] ([kaldi][1467])  │[1468]           │erUnsortedArchiveImpl]│dAligner::TupleHas
                     │([kaldi::nnet3][1│[1470] ([kaldi][1471])│h][1472]          
                     │469])            │                      │([kaldi][1473])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[ConstantFunctionComp│[HashList::HashBu│[NnetComputeProbOption│[RandomComponent][│[LatticePhoneAlign
onent][1474]         │cket][1476]      │s][1478]              │1480]             │er::TupleHash][148
([kaldi::nnet3][1475]│([kaldi][1477])  │([kaldi::nnet3][1479])│([kaldi::nnet2][14│2] ([kaldi][1483])
)                    │                 │                      │81])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConstantSumDescripto│[HashList][1486] │[NnetComputer][1488]  │[RandomComponent][│[LatticeWordAligne
r][1484]             │([kaldi][1487])  │([kaldi::nnet2][1489])│1490]             │r::TupleHash][1492
([kaldi::nnet3][1485]│                 │                      │([kaldi::nnet3][14│] ([kaldi][1493]) 
)                    │                 │                      │91])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConstArpaLm][1494]  │[HiddenSoftmax][1│[NnetComputer][1498]  │[RandomizerMask][1│[TwvMetrics][1502]
([kaldi][1495])      │496]             │([kaldi::nnet3][1499])│500]              │([kaldi][1503])   
                     │([kaldi::nnet1][1│                      │([kaldi::nnet1][15│                  
                     │497])            │                      │01])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConstArpaLmBuilder][│[HmmCacheHash][15│[NnetComputerFromEg][1│[RandomState][1510│[TwvMetricsOptions
1504] ([kaldi][1505])│06]              │508]                  │] ([kaldi][1511]) │][1512]           
                     │([kaldi][1507])  │([kaldi::nnet3][1509])│                  │([kaldi][1513])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConstArpaLmDetermini│[HmmTopology::Hmm│[NnetDataRandomizerOpt│[Rbm][1520]       │[TwvMetricsStats][
sticFst][1514]       │State][1516]     │ions][1518]           │([kaldi::nnet1][15│1522]             
([kaldi][1515])      │([kaldi][1517])  │([kaldi::nnet1][1519])│21])              │([kaldi][1523])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConstIntegerSet][152│[HmmTopology][152│[NnetDiscriminativeCom│[RbmBase][1530]   │u                 
4] ([kaldi][1525])   │6]               │puteObjf][1528]       │([kaldi::nnet1][15│                  
                     │([kaldi][1527])  │([kaldi::nnet3][1529])│31])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[ContextDependency][1│[HtkHeader][1534]│[NnetDiscriminativeExa│[RbmTrainOptions][
532] ([kaldi][1533]) │([kaldi][1535])  │mple][1536]           │1538]             
                     │                 │([kaldi::nnet3][1537])│([kaldi::nnet1][15
                     │                 │                      │39])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[ContextDependencyInt│[HtkMatrixHolder]│[NnetDiscriminativeExa│[RecognizedWord][1│[UbmClusteringOpti
erface][1540]        │[1542]           │mpleStructureCompare][│546]              │ons][1548]        
([kaldi][1541])      │([kaldi][1543])  │1544]                 │([kaldi][1547])   │([kaldi][1549])   
                     │                 │([kaldi::nnet3][1545])│                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[Convolutional1dCompo│[HTransducerConfi│[NnetDiscriminativeExa│[RectifiedLinearCo│[UnweightedNgramFs
nent][1550]          │g][1552]         │mpleStructureHasher][1│mponent][1556]    │t][1558]          
([kaldi::nnet2][1551]│([kaldi][1553])  │554]                  │([kaldi::nnet2][15│([fst][1559])     
)                    │                 │([kaldi::nnet3][1555])│57])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConvolutionalCompone│i                │[NnetDiscriminativeOpt│[RectifiedLinearCo│[UpdatableComponen
nt][1560]            │                 │ions][1562]           │mponent][1564]    │t][1566]          
([kaldi::nnet1][1561]│                 │([kaldi::nnet3][1563])│([kaldi::nnet3][15│([kaldi::nnet3][15
)                    │                 │                      │65])              │67])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[ConvolutionComponent│[NnetDiscriminati│[RecurrentComponent][1│[UpdatableComponen
][1568]              │veStats][1570]   │572]                  │t][1574]          
([kaldi::nnet3][1569]│([kaldi::nnet2][1│([kaldi::nnet1][1573])│([kaldi::nnet1][15
)                    │571])            │                      │75])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[ConvolutionComputati│[IdentityFunction│[NnetDiscriminativeSup│[RecyclingVector][│[UpdatableComponen
on][1576]            │][1578]          │ervision][1580]       │1582]             │t][1584]          
([kaldi::nnet3::time_│([fst][1579])    │([kaldi::nnet3][1581])│([kaldi][1583])   │([kaldi::nnet2][15
height_convolution][1│                 │                      │                  │85])              
577])                │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConvolutionComputati│[ImageAugmentatio│[NnetDiscriminativeTra│[RefineClusterer][│[UpdatePhoneVector
onIo][1586]          │nConfig][1588]   │iner][1590]           │1592]             │sClass][1594]     
([kaldi::nnet3::time_│([kaldi::nnet3][1│([kaldi::nnet3][1591])│([kaldi][1593])   │([kaldi][1595])   
height_convolution][1│589])            │                      │                  │                  
587])                │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConvolutionComputati│[ImplToFst][1598]│[NnetDiscriminativeUpd│[RefineClustersOpt│[UpdateWClass][160
onOptions][1596]     │                 │ateOptions][1599]     │ions][1601]       │3] ([kaldi][1604])
([kaldi::nnet3::time_│                 │([kaldi::nnet2][1600])│([kaldi][1602])   │                  
height_convolution][1│                 │                      │                  │                  
597])                │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConvolutionModel][16│[Index][1607]    │[NnetDiscriminativeUpd│[RegressionTree][1│[NnetBatchInferenc
05]                  │([kaldi::nnet3][1│ater][1609]           │611]              │e::UtteranceInfo][
([kaldi::nnet3::time_│608])            │([kaldi::nnet2][1610])│([kaldi][1612])   │1613]             
height_convolution][1│                 │                      │                  │([kaldi::nnet3][16
606])                │                 │                      │                  │14])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[ConvolutionComputati│[IndexHasher][161│[NnetEnsembleTrainer][│[RegtreeFmllrDiagG│[NnetBatchDecoder:
on::ConvolutionStep][│7]               │1619]                 │mm][1621]         │:UtteranceInput][1
1615]                │([kaldi::nnet3][1│([kaldi::nnet2][1620])│([kaldi][1622])   │623]              
([kaldi::nnet3::time_│618])            │                      │                  │([kaldi::nnet3][16
height_convolution][1│                 │                      │                  │24])              
616])                │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CopyComponent][1625]│[IndexLessNxt][16│[NnetEnsembleTrainerCo│[RegtreeFmllrDiagG│[NnetBatchDecoder:
([kaldi::nnet1][1626]│27]              │nfig][1629]           │mmAccs][1631]     │:UtteranceOutput][
)                    │([kaldi::nnet3][1│([kaldi::nnet2][1630])│([kaldi][1632])   │1633]             
                     │628])            │                      │                  │([kaldi::nnet3][16
                     │                 │                      │                  │34])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CountStats][1635]   │[IndexSet][1637] │[NnetExample][1639]   │[RegtreeFmllrOptio│[UtteranceSplitter
([kaldi][1636])      │([kaldi::nnet3][1│([kaldi::nnet2][1640])│ns][1641]         │][1643]           
                     │638])            │                      │([kaldi][1642])   │([kaldi::nnet3][16
                     │                 │                      │                  │44])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CovarianceStats][164│[IndexVectorHashe│[NnetExample][1649]   │[RegtreeMllrDiagGm│v                 
5] ([kaldi][1646])   │r][1647]         │([kaldi::nnet3][1650])│m][1651]          │                  
                     │([kaldi::nnet3][1│                      │([kaldi][1652])   │                  
                     │648])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[CRnnLM][1653]       │[Input][1655]    │[NnetExampleBackground│[RegtreeMllrDiagGm
([rnnlm][1654])      │([kaldi][1656])  │Reader][1657]         │mAccs][1659]      
                     │                 │([kaldi::nnet2][1658])│([kaldi][1660])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[CuAllocatorOptions][│[InputImplBase][1│[NnetExampleStructureC│[RegtreeMllrOption│[VadEnergyOptions]
1661] ([kaldi][1662])│663]             │ompare][1665]         │s][1667]          │[1669]            
                     │([kaldi][1664])  │([kaldi::nnet3][1666])│([kaldi][1668])   │([kaldi][1670])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuArray][1671]      │[Int32AndFloat][1│[NnetExampleStructureH│[RemoveEpsLocalCla│[VariableMergingOp
([kaldi][1672])      │673]             │asher][1675]          │ss][1677]         │timizer][1679]    
                     │([kaldi][1674])  │([kaldi::nnet3][1676])│([fst][1678])     │([kaldi::nnet3][16
                     │                 │                      │                  │80])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuArrayBase][1681]  │[Int32IsZero][168│[NnetFixConfig][1685] │[RemoveSomeInputSy│[Vector][1689]    
([kaldi][1682])      │3]               │([kaldi::nnet2][1686])│mbolsMapper][1687]│([kaldi][1690])   
                     │([kaldi][1684])  │                      │([fst][1688])     │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuBlockMatrix][1691]│[Int32Pair][1693]│[NnetGenerationOptions│[RepeatedAffineCom│[VectorBase][1698]
([kaldi][1692])      │                 │][1694]               │ponent][1696]     │([kaldi][1699])   
                     │                 │([kaldi::nnet3][1695])│([kaldi::nnet3][16│                  
                     │                 │                      │97])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuBlockMatrixData_][│[Interval][1701] │[NnetInferenceTask][17│[ReplaceIndexForwa│[VectorClusterable
1700]                │([kaldi][1702])  │03]                   │rdingDescriptor][1│][1707]           
                     │                 │([kaldi::nnet3][1704])│705]              │([kaldi][1708])   
                     │                 │                      │([kaldi::nnet3][17│                  
                     │                 │                      │06])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuCompressedMatrix][│[ExampleMergingCo│[NnetIo][1713]        │[Rescale][1715]   │[StringRepository:
1709] ([kaldi][1710])│nfig::IntSet][171│([kaldi::nnet3][1714])│([kaldi::nnet1][17│:VectorEqual][1717
                     │1]               │                      │16])              │] ([fst][1718])   
                     │([kaldi::nnet3][1│                      │                  │                  
                     │712])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuCompressedMatrixBa│[InverseContextFs│[NnetIoStructureCompar│[RestrictedAttenti│[VectorFstToKwsLex
se][1719]            │t][1721]         │e][1723]              │onComponent][1725]│icographicFstMappe
([kaldi][1720])      │([fst][1722])    │([kaldi::nnet3][1724])│([kaldi::nnet3][17│r][1727]          
                     │                 │                      │26])              │([kaldi][1728])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuMatrix][1729]     │[InverseLeftBipho│[NnetIoStructureHasher│[ProfileStats::Rev│[VectorFstTplHolde
([kaldi][1730])      │neContextFst][173│][1733]               │erseSecondComparat│r][1737]          
                     │1] ([fst][1732]) │([kaldi::nnet3][1734])│or][1735]         │([fst][1738])     
                     │                 │                      │([kaldi][1736])   │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuMatrixBase][1739] │[IoSpecification]│[NnetLdaStatsAccumulat│[ReweightPlusDefau│[VectorHasher][174
([kaldi][1740])      │[1741]           │or][1743]             │lt][1745]         │7] ([kaldi][1748])
                     │([kaldi::nnet3][1│([kaldi::nnet3][1744])│([fst][1746])     │                  
                     │742])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuPackedMatrix][1749│[IoSpecificationH│[NnetLimitRankOpts][17│[ReweightPlusLogAr│[StringRepository:
] ([kaldi][1750])    │asher][1751]     │53]                   │c][1755]          │:VectorKey][1757] 
                     │([kaldi::nnet3][1│([kaldi::nnet2][1754])│([fst][1756])     │([fst][1758])     
                     │752])            │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuRand][1759]       │[IvectorEstimatio│[NnetMixupConfig][1763│[RnnlmDeterministi│[VectorRandomizer]
([kaldi][1760])      │nOptions][1761]  │]                     │cFst][1765]       │[1767]            
                     │([kaldi][1762])  │([kaldi::nnet2][1764])│([kaldi][1766])   │([kaldi::nnet1][17
                     │                 │                      │                  │68])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuSparseMatrix][1769│[IvectorExtractor│[NnetOnlineComputer][1│[RoundingForwardin│[vocab_word][1777]
] ([kaldi][1770])    │][1771]          │773]                  │gDescriptor][1775]│([rnnlm][1778])   
                     │([kaldi][1772])  │([kaldi::nnet2][1774])│([kaldi::nnet3][17│                  
                     │                 │                      │76])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuSpMatrix][1779]   │[IvectorExtractor│[NnetOptimizeOptions][│[RowOpsSplitter][1│w                 
([kaldi][1780])      │ComputeDerivedVar│1783]                 │785]              │                  
                     │sClass][1781]    │([kaldi::nnet3][1784])│([kaldi::nnet3][17│                  
                     │([kaldi][1782])  │                      │86])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[CuSubArray][1787]   │[IvectorExtractor│[NnetRescaleConfig][17│[RspecifierOptions
([kaldi][1788])      │EstimationOptions│91]                   │][1793]           
                     │][1789]          │([kaldi::nnet2][1792])│([kaldi][1794])   
                     │([kaldi][1790])  │                      │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[CuSubMatrix][1795]  │[IvectorExtractor│[NnetRescaler][1799]  │[TaskSequencer::Ru│[WaveData][1803]  
([kaldi][1796])      │Options][1797]   │([kaldi::nnet2][1800])│nTaskArgsList][180│([kaldi][1804])   
                     │([kaldi][1798])  │                      │1] ([kaldi][1802])│                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuSubVector][1805]  │[IvectorExtractor│[NnetShrinkConfig][180│s                 │[WaveHeaderReadGof
([kaldi][1806])      │Stats][1807]     │9]                    │                  │er][1811]         
                     │([kaldi][1808])  │([kaldi::nnet2][1810])│                  │([kaldi][1812])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[CuTpMatrix][1813]   │[IvectorExtractor│[NnetSimpleComputation│[WaveHolder][1819]
([kaldi][1814])      │StatsOptions][181│Options][1817]        │([kaldi][1820])   
                     │5]               │([kaldi::nnet3][1818])│                  
                     │([kaldi][1816])  │                      │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[CuValue][1821]      │[IvectorExtractor│[NnetSimpleLoopedCompu│[ScalarClusterable│[WaveInfo][1829]  
([kaldi][1822])      │UpdateProjectionC│tationOptions][1825]  │][1827]           │([kaldi][1830])   
                     │lass][1823]      │([kaldi::nnet3][1826])│([kaldi][1828])   │                  
                     │([kaldi][1824])  │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuVector][1831]     │[IvectorExtractor│[NnetSimpleTrainerConf│[ScaleAndOffsetCom│[WaveInfoHolder][1
([kaldi][1832])      │UpdateWeightClass│ig][1835]             │ponent][1837]     │839]              
                     │][1833]          │([kaldi::nnet2][1836])│([kaldi::nnet3][18│([kaldi][1840])   
                     │([kaldi][1834])  │                      │38])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[CuVectorBase][1841] │[IvectorExtractor│[NnetStats][1845]     │[ScaleComponent][1│[WordAlignedLattic
([kaldi][1842])      │UtteranceStats][1│([kaldi::nnet2][1846])│847]              │eTester][1849]    
                     │843]             │                      │([kaldi::nnet2][18│([kaldi][1850])   
                     │([kaldi][1844])  │                      │48])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
d                    │[IvectorExtractTa│[NnetStatsConfig][1853│[ScaleDeterministi│[WordAlignLatticeL
                     │sk][1851]        │]                     │cOnDemandFst][1855│exiconInfo][1857] 
                     │([kaldi][1852])  │([kaldi::nnet2][1854])│] ([fst][1856])   │([kaldi][1858])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[IvectorTask][1859]  │[NnetTrainer][186│[Semaphore][1863]     │[WordAlignLatticeL
([kaldi][1860])      │1]               │([kaldi][1864])       │exiconOpts][1865] 
                     │([kaldi::nnet3][1│                      │([kaldi][1866])   
                     │862])            │                      │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[DctComponent][1867] │k                │[NnetTrainerOptions][1│[SentenceAveraging│[WordBoundaryInfo]
([kaldi::nnet2][1868]│                 │869]                  │Component][1871]  │[1873]            
)                    │                 │([kaldi::nnet3][1870])│([kaldi::nnet1][18│([kaldi][1874])   
                     │                 │                      │72])              │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[DecisionTreeSplitter│[NnetTrainOptions│[SequenceTransform][18│[WordBoundaryInfoN
][1875]              │][1877]          │79]                   │ewOpts][1881]     
([kaldi][1876])      │([kaldi::nnet1][1│([kaldi::differentiabl│([kaldi][1882])   
                     │878])            │e_transform][1880])   │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[DecodableAmDiagGmm][│[KaldiCompileTime│[NnetUpdater][1886]   │[SequentialTableRe│[WordBoundaryInfoO
1883] ([kaldi][1884])│Assert][1885]    │([kaldi::nnet2][1887])│ader][1888]       │pts][1890]        
                     │                 │                      │([kaldi][1889])   │([kaldi][1891])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[DecodableAmDiagGmmRe│[KaldiCompileTime│[NnetWidenConfig][1895│[SequentialTableRe│[ConstArpaLmBuilde
gtreeFmllr][1892]    │Assert< true     │]                     │aderArchiveImpl][1│r::WordsAndLmState
([kaldi][1893])      │>][1894]         │([kaldi::nnet2][1896])│897]              │PairLessThan][1899
                     │                 │                      │([kaldi][1898])   │] ([kaldi][1900]) 
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[DecodableAmDiagGmmRe│[KaldiFatalError]│[TreeClusterer::Node][│[SequentialTableRe│[WspecifierOptions
gtreeMllr][1901]     │[1903]           │1905] ([kaldi][1906]) │aderBackgroundImpl│][1909]           
([kaldi][1902])      │([kaldi][1904])  │                      │][1907]           │([kaldi][1910])   
                     │                 │                      │([kaldi][1908])   │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[DecodableAmDiagGmmSc│[KaldiObjectHolde│[NonlinearComponent][1│[SequentialTableRe│x                 
aled][1911]          │r][1913]         │915]                  │aderImplBase][1917│                  
([kaldi][1912])      │([kaldi][1914])  │([kaldi::nnet2][1916])│] ([kaldi][1918]) │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┴──────────────────
[DecodableAmDiagGmmUn│[KaldiRnnlmWrappe│[NonlinearComponent][1│[SequentialTableRe
mapped][1919]        │r][1921]         │923]                  │aderScriptImpl][19
([kaldi][1920])      │([kaldi][1922])  │([kaldi::nnet3][1924])│25]               
                     │                 │                      │([kaldi][1926])   
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┬
[DecodableAmNnet][192│[KaldiRnnlmWrappe│[NoOpComponent][1931] │[Sgmm2FmllrConfig]│[Xent][1935]      
7]                   │rOpts][1929]     │([kaldi::nnet3][1932])│[1933]            │([kaldi::nnet1][19
([kaldi::nnet2][1928]│([kaldi][1930])  │                      │([kaldi][1934])   │36])              
)                    │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[DecodableAmNnetLoope│[Component::key_v│[NoOpTransform][1941] │[Sgmm2FmllrGlobalP│[BatchedXvectorCom
dOnline][1937]       │alue][1939]      │([kaldi::differentiabl│arams][1943]      │puter::XvectorTask
([kaldi::nnet3][1938]│([kaldi::nnet1][1│e_transform][1942])   │([kaldi][1944])   │][1945]           
)                    │940])            │                      │                  │([kaldi::nnet3][19
                     │                 │                      │                  │46])              
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[DecodableAmNnetParal│[KlHmm][1949]    │[OnlineProcessPitch::N│[Sgmm2GauPost][195│                  
lel][1947]           │([kaldi::nnet1][1│ormalizationStats][195│3] ([kaldi][1954])│                  
([kaldi::nnet2][1948]│950])            │1] ([kaldi][1952])    │                  │                  
)                    │                 │                      │                  │                  
─────────────────────┼─────────────────┼──────────────────────┼──────────────────┼──────────────────
[DecodableAmNnetSimpl│[kMarkerMap][1957│[NormalizeComponent][1│[Sgmm2GauPostEleme│                  
e][1955]             │]                │958]                  │nt][1960]         │                  
([kaldi::nnet3][1956]│                 │([kaldi::nnet3][1959])│([kaldi][1961])   │                  
)                    │                 │                      │                  │                  
─────────────────────┴─────────────────┴──────────────────────┴──────────────────┴──────────────────
[a][1962] | [b][1963] | [c][1964] | [d][1965] | [e][1966] | [f][1967] | [g][1968] | [h][1969] |
[i][1970] | [k][1971] | [l][1972] | [m][1973] | [n][1974] | [o][1975] | [p][1976] | [q][1977] |
[r][1978] | [s][1979] | [t][1980] | [u][1981] | [v][1982] | [w][1983] | [x][1984]

* Generated by [ [doxygen]][1985] 1.8.13

[1]: http://kaldi-asr.org/
[2]: #letter_a
[3]: #letter_b
[4]: #letter_c
[5]: #letter_d
[6]: #letter_e
[7]: #letter_f
[8]: #letter_g
[9]: #letter_h
[10]: #letter_i
[11]: #letter_k
[12]: #letter_l
[13]: #letter_m
[14]: #letter_n
[15]: #letter_o
[16]: #letter_p
[17]: #letter_q
[18]: #letter_r
[19]: #letter_s
[20]: #letter_t
[21]: #letter_u
[22]: #letter_v
[23]: #letter_w
[24]: #letter_x
[25]: classkaldi_1_1nnet3_1_1DecodableAmNnetSimpleLooped.html
[26]: namespacekaldi_1_1nnet3.html
[27]: classkaldi_1_1KwsAlignment.html
[28]: namespacekaldi.html
[29]: classkaldi_1_1nnet2_1_1NormalizeComponent.html
[30]: namespacekaldi_1_1nnet2.html
[31]: structkaldi_1_1Sgmm2GselectConfig.html
[32]: namespacekaldi.html
[33]: classkaldi_1_1nnet3_1_1DecodableAmNnetSimpleParallel.html
[34]: namespacekaldi_1_1nnet3.html
[35]: structkaldi_1_1kws__internal_1_1KwScoreStats.html
[36]: namespacekaldi_1_1kws__internal.html
[37]: classkaldi_1_1NumberIstream.html
[38]: namespacekaldi.html
[39]: structkaldi_1_1Sgmm2LikelihoodCache.html
[40]: namespacekaldi.html
[41]: structkaldi_1_1nnet3_1_1Access.html
[42]: namespacekaldi_1_1nnet3.html
[43]: classkaldi_1_1DecodableAmSgmm2.html
[44]: namespacekaldi.html
[45]: classkaldi_1_1KwsProductFstToKwsLexicographicFstMapper.html
[46]: namespacekaldi.html
[47]: structkaldi_1_1Sgmm2PerFrameDerivedVars.html
[48]: namespacekaldi.html
[49]: classkaldi_1_1AccumAmDiagGmm.html
[50]: namespacekaldi.html
[51]: classkaldi_1_1DecodableAmSgmm2Scaled.html
[52]: namespacekaldi.html
[53]: classkaldi_1_1KwsTerm.html
[54]: namespacekaldi.html
[55]: classkaldi_1_1Sgmm2PerSpkDerivedVars.html
[56]: namespacekaldi.html
[57]: classkaldi_1_1AccumDiagGmm.html
[58]: namespacekaldi.html
[59]: classkaldi_1_1DecodableDiagGmmScaledOnline.html
[60]: namespacekaldi.html
[61]: classkaldi_1_1KwsTermsAligner.html
[62]: namespacekaldi.html
[63]: structkaldi_1_1nnet3_1_1ObjectiveFunctionInfo.html
[64]: namespacekaldi_1_1nnet3.html
[65]: classkaldi_1_1Sgmm2Project.html
[66]: namespacekaldi.html
[67]: classkaldi_1_1AccumFullGmm.html
[68]: namespacekaldi.html
[69]: classkaldi_1_1DecodableInterface.html
[70]: namespacekaldi.html
[71]: structkaldi_1_1KwsTermsAlignerOptions.html
[72]: namespacekaldi.html
[73]: classkaldi_1_1OfflineFeatureTpl.html
[74]: namespacekaldi.html
[75]: structkaldi_1_1Sgmm2SplitSubstatesConfig.html
[76]: namespacekaldi.html
[77]: classkaldi_1_1AccumulateMultiThreadedClass.html
[78]: namespacekaldi.html
[79]: classkaldi_1_1DecodableMapped.html
[80]: namespacekaldi.html
[81]: classkaldi_1_1kws__internal_1_1KwTermEqual.html
[82]: namespacekaldi_1_1kws__internal.html
[83]: structkaldi_1_1nnet3_1_1time__height__convolution_1_1ConvolutionModel_1_1Offset.html
[84]: namespacekaldi_1_1nnet3_1_1time__height__convolution.html
[85]: classkaldi_1_1ShiftedDeltaFeatures.html
[86]: namespacekaldi.html
[87]: structkaldi_1_1AccumulateTreeStatsInfo.html
[88]: namespacekaldi.html
[89]: classkaldi_1_1DecodableMatrixMapped.html
[90]: namespacekaldi.html
[91]: classkaldi_1_1kws__internal_1_1KwTermLower.html
[92]: namespacekaldi_1_1kws__internal.html
[93]: classkaldi_1_1OffsetFileInputImpl.html
[94]: namespacekaldi.html
[95]: structkaldi_1_1ShiftedDeltaFeaturesOptions.html
[96]: namespacekaldi.html
[97]: structkaldi_1_1AccumulateTreeStatsOptions.html
[98]: namespacekaldi.html
[99]: classkaldi_1_1DecodableMatrixMappedOffset.html
[100]: namespacekaldi.html
[101]: classkaldi_1_1nnet3_1_1OffsetForwardingDescriptor.html
[102]: namespacekaldi_1_1nnet3.html
[103]: classkaldi_1_1nnet1_1_1Sigmoid.html
[104]: namespacekaldi_1_1nnet1.html
[105]: structkaldi_1_1ActivePath.html
[106]: namespacekaldi.html
[107]: classkaldi_1_1DecodableMatrixScaled.html
[108]: namespacekaldi.html
[109]: classkaldi_1_1OnlineAppendFeature.html
[110]: namespacekaldi.html
[111]: classkaldi_1_1nnet2_1_1SigmoidComponent.html
[112]: namespacekaldi_1_1nnet2.html
[113]: classkaldi_1_1nnet2_1_1AdditiveNoiseComponent.html
[114]: namespacekaldi_1_1nnet2.html
[115]: classkaldi_1_1DecodableMatrixScaledMapped.html
[116]: namespacekaldi.html
[117]: structkaldi_1_1LatticeArcRecord.html
[118]: namespacekaldi.html
[119]: classkaldi_1_1OnlineAudioSourceItf.html
[120]: namespacekaldi.html
[121]: classkaldi_1_1nnet3_1_1SigmoidComponent.html
[122]: namespacekaldi_1_1nnet3.html
[123]: classkaldi_1_1nnet1_1_1AddShift.html
[124]: namespacekaldi_1_1nnet1.html
[125]: classkaldi_1_1nnet2_1_1DecodableNnet2Online.html
[126]: namespacekaldi_1_1nnet2.html
[127]: classkaldi_1_1LatticeBiglmFasterDecoder.html
[128]: namespacekaldi.html
[129]: classkaldi_1_1OnlineBaseFeature.html
[130]: namespacekaldi.html
[131]: classkaldi_1_1SimpleDecoder.html
[132]: namespacekaldi.html
[133]: classkaldi_1_1nnet2_1_1AffineComponent.html
[134]: namespacekaldi_1_1nnet2.html
[135]: structkaldi_1_1nnet2_1_1DecodableNnet2OnlineOptions.html
[136]: namespacekaldi_1_1nnet2.html
[137]: classfst_1_1LatticeDeterminizer.html
[138]: namespacefst.html
[139]: classkaldi_1_1OnlineCacheFeature.html
[140]: namespacekaldi.html
[141]: classkaldi_1_1nnet3_1_1SimpleForwardingDescriptor.html
[142]: namespacekaldi_1_1nnet3.html
[143]: classkaldi_1_1nnet3_1_1AffineComponent.html
[144]: namespacekaldi_1_1nnet3.html
[145]: classkaldi_1_1nnet3_1_1DecodableNnetLoopedOnline.html
[146]: namespacekaldi_1_1nnet3.html
[147]: classfst_1_1LatticeDeterminizerPruned.html
[148]: namespacefst.html
[149]: classkaldi_1_1OnlineCacheInput.html
[150]: namespacekaldi.html
[151]: classkaldi_1_1differentiable__transform_1_1SimpleMeanTransform.html
[152]: namespacekaldi_1_1differentiable__transform.html
[153]: classkaldi_1_1nnet2_1_1AffineComponentPreconditioned.html
[154]: namespacekaldi_1_1nnet2.html
[155]: classkaldi_1_1nnet3_1_1DecodableNnetLoopedOnlineBase.html
[156]: namespacekaldi_1_1nnet3.html
[157]: structkaldi_1_1LatticeFasterDecoderConfig.html
[158]: namespacekaldi.html
[159]: classkaldi_1_1OnlineCmnInput.html
[160]: namespacekaldi.html
[161]: structkaldi_1_1nnet3_1_1SimpleObjectiveInfo.html
[162]: namespacekaldi_1_1nnet3.html
[163]: classkaldi_1_1nnet2_1_1AffineComponentPreconditionedOnline.html
[164]: namespacekaldi_1_1nnet2.html
[165]: classkaldi_1_1nnet3_1_1DecodableNnetSimple.html
[166]: namespacekaldi_1_1nnet3.html
[167]: classkaldi_1_1LatticeFasterDecoderTpl.html
[168]: namespacekaldi.html
[169]: classkaldi_1_1OnlineCmvn.html
[170]: namespacekaldi.html
[171]: classkaldi_1_1SimpleOptions.html
[172]: namespacekaldi.html
[173]: classkaldi_1_1nnet1_1_1AffineTransform.html
[174]: namespacekaldi_1_1nnet1.html
[175]: classkaldi_1_1nnet3_1_1DecodableNnetSimpleLooped.html
[176]: namespacekaldi_1_1nnet3.html
[177]: classkaldi_1_1LatticeFasterOnlineDecoderTpl.html
[178]: namespacekaldi.html
[179]: structkaldi_1_1OnlineCmvnOptions.html
[180]: namespacekaldi.html
[181]: classkaldi_1_1nnet1_1_1SimpleSentenceAveragingComponent.html
[182]: namespacekaldi_1_1nnet1.html
[183]: classkaldi_1_1AffineXformStats.html
[184]: namespacekaldi.html
[185]: classkaldi_1_1nnet3_1_1DecodableNnetSimpleLoopedInfo.html
[186]: namespacekaldi_1_1nnet3.html
[187]: classkaldi_1_1LatticeHolder.html
[188]: namespacekaldi.html
[189]: structkaldi_1_1OnlineCmvnState.html
[190]: namespacekaldi.html
[191]: classkaldi_1_1nnet3_1_1SimpleSumDescriptor.html
[192]: namespacekaldi_1_1nnet3.html
[193]: classkaldi_1_1AgglomerativeClusterer.html
[194]: namespacekaldi.html
[195]: classkaldi_1_1DecodableSum.html
[196]: namespacekaldi.html
[197]: structkaldi_1_1LatticeIncrementalDecoderConfig.html
[198]: namespacekaldi.html
[199]: classkaldi_1_1OnlineDecodableDiagGmmScaled.html
[200]: namespacekaldi.html
[201]: structkaldi_1_1FmllrRawAccs_1_1SingleFrameStats.html
[202]: namespacekaldi.html
[203]: structkaldi_1_1AhcCluster.html
[204]: namespacekaldi.html
[205]: classkaldi_1_1DecodableSumScaled.html
[206]: namespacekaldi.html
[207]: classkaldi_1_1LatticeIncrementalDecoderTpl.html
[208]: namespacekaldi.html
[209]: classkaldi_1_1OnlineDeltaFeature.html
[210]: namespacekaldi.html
[211]: structkaldi_1_1FmllrDiagGmmAccs_1_1SingleFrameStats.html
[212]: namespacekaldi.html
[213]: structkaldi_1_1AlignConfig.html
[214]: namespacekaldi.html
[215]: structDecodeInfo.html
[216]: classkaldi_1_1LatticeIncrementalDeterminizer.html
[217]: namespacekaldi.html
[218]: classkaldi_1_1OnlineDeltaInput.html
[219]: namespacekaldi.html
[220]: structkaldi_1_1nnet3_1_1RowOpsSplitter_1_1SingleSplitInfo.html
[221]: namespacekaldi_1_1nnet3.html
[222]: structkaldi_1_1AlignedTermsPair.html
[223]: namespacekaldi.html
[224]: classkaldi_1_1DecodeUtteranceLatticeFasterClass.html
[225]: namespacekaldi.html
[226]: classkaldi_1_1LatticeIncrementalOnlineDecoderTpl.html
[227]: namespacekaldi.html
[228]: structkaldi_1_1OnlineEndpointConfig.html
[229]: namespacekaldi.html
[230]: classkaldi_1_1SingleUtteranceGmmDecoder.html
[231]: namespacekaldi.html
[232]: classkaldi_1_1AmDiagGmm.html
[233]: namespacekaldi.html
[234]: classkaldi_1_1DeltaFeatures.html
[235]: namespacekaldi.html
[236]: structkaldi_1_1discriminative_1_1DiscriminativeSupervisionSplitter_1_1LatticeInfo.html
[237]: namespacekaldi_1_1discriminative.html
[238]: structkaldi_1_1OnlineEndpointRule.html
[239]: namespacekaldi.html
[240]: classkaldi_1_1SingleUtteranceNnet2Decoder.html
[241]: namespacekaldi.html
[242]: classkaldi_1_1nnet2_1_1AmNnet.html
[243]: namespacekaldi_1_1nnet2.html
[244]: structkaldi_1_1DeltaFeaturesOptions.html
[245]: namespacekaldi.html
[246]: classkaldi_1_1LatticeLexiconWordAligner.html
[247]: namespacekaldi.html
[248]: classkaldi_1_1OnlineFasterDecoder.html
[249]: namespacekaldi.html
[250]: classkaldi_1_1SingleUtteranceNnet2DecoderThreaded.html
[251]: namespacekaldi.html
[252]: classkaldi_1_1nnet3_1_1AmNnetSimple.html
[253]: namespacekaldi_1_1nnet3.html
[254]: classkaldi_1_1nnet3_1_1DerivativeTimeLimiter.html
[255]: namespacekaldi_1_1nnet3.html
[256]: classkaldi_1_1LatticePhoneAligner.html
[257]: namespacekaldi.html
[258]: structkaldi_1_1OnlineFasterDecoderOpts.html
[259]: namespacekaldi.html
[260]: classkaldi_1_1SingleUtteranceNnet3DecoderTpl.html
[261]: namespacekaldi.html
[262]: classkaldi_1_1AmSgmm2.html
[263]: namespacekaldi.html
[264]: classkaldi_1_1nnet3_1_1Descriptor.html
[265]: namespacekaldi_1_1nnet3.html
[266]: classkaldi_1_1LatticeReader.html
[267]: namespacekaldi.html
[268]: classkaldi_1_1OnlineFeatInputItf.html
[269]: namespacekaldi.html
[270]: classkaldi_1_1SingleUtteranceNnet3IncrementalDecoderTpl.html
[271]: namespacekaldi.html
[272]: structkaldi_1_1nnet3_1_1Analyzer.html
[273]: namespacekaldi_1_1nnet3.html
[274]: classfst_1_1DeterministicOnDemandFst.html
[275]: namespacefst.html
[276]: classkaldi_1_1LatticeSimpleDecoder.html
[277]: namespacekaldi.html
[278]: classkaldi_1_1OnlineFeatureInterface.html
[279]: namespacekaldi.html
[280]: structkaldi_1_1SlidingWindowCmnOptions.html
[281]: namespacekaldi.html
[282]: classkaldi_1_1differentiable__transform_1_1AppendTransform.html
[283]: namespacekaldi_1_1differentiable__transform.html
[284]: structfst_1_1DeterminizeLatticeOptions.html
[285]: namespacefst.html
[286]: structkaldi_1_1LatticeSimpleDecoderConfig.html
[287]: namespacekaldi.html
[288]: classkaldi_1_1OnlineFeatureMatrix.html
[289]: namespacekaldi.html
[290]: classkaldi_1_1nnet2_1_1SoftHingeComponent.html
[291]: namespacekaldi_1_1nnet2.html
[292]: classkaldi_1_1ArbitraryResample.html
[293]: namespacekaldi.html
[294]: structfst_1_1DeterminizeLatticePhonePrunedOptions.html
[295]: namespacefst.html
[296]: structkaldi_1_1PrunedCompactLatticeComposer_1_1LatticeStateInfo.html
[297]: namespacekaldi.html
[298]: structkaldi_1_1OnlineFeatureMatrixOptions.html
[299]: namespacekaldi.html
[300]: classkaldi_1_1nnet1_1_1Softmax.html
[301]: namespacekaldi_1_1nnet1.html
[302]: structkaldi_1_1MinimumBayesRisk_1_1Arc.html
[303]: namespacekaldi.html
[304]: structfst_1_1DeterminizeLatticePrunedOptions.html
[305]: namespacefst.html
[306]: classfst_1_1LatticeStringRepository.html
[307]: namespacefst.html
[308]: classkaldi_1_1OnlineFeaturePipeline.html
[309]: namespacekaldi.html
[310]: classkaldi_1_1nnet2_1_1SoftmaxComponent.html
[311]: namespacekaldi_1_1nnet2.html
[312]: structfst_1_1GrammarFstPreparer_1_1ArcCategory.html
[313]: namespacefst.html
[314]: classkaldi_1_1DeterminizeLatticeTask.html
[315]: namespacekaldi.html
[316]: classfst_1_1LatticeToStdMapper.html
[317]: namespacefst.html
[318]: structkaldi_1_1OnlineFeaturePipelineCommandLineConfig.html
[319]: namespacekaldi.html
[320]: classkaldi_1_1nnet3_1_1SoftmaxComponent.html
[321]: namespacekaldi_1_1nnet3.html
[322]: classfst_1_1ArcIterator_3_01GrammarFst_01_4.html
[323]: namespacefst.html
[324]: classfst_1_1DeterminizerStar.html
[325]: namespacefst.html
[326]: classfst_1_1LatticeWeightTpl.html
[327]: namespacefst.html
[328]: structkaldi_1_1OnlineFeaturePipelineConfig.html
[329]: namespacekaldi.html
[330]: structkaldi_1_1SolverOptions.html
[331]: namespacekaldi.html
[332]: classfst_1_1ArcIterator_3_01TrivialFactorWeightFst_3_01A_00_01F_01_4_01_4.html
[333]: namespacefst.html
[334]: classfst_1_1DfsOrderVisitor.html
[335]: namespacefst.html
[336]: classkaldi_1_1LatticeWordAligner.html
[337]: namespacekaldi.html
[338]: classkaldi_1_1OnlineFeInput.html
[339]: namespacekaldi.html
[340]: classkaldi_1_1SparseMatrix.html
[341]: namespacekaldi.html
[342]: classkaldi_1_1ArcPosteriorComputer.html
[343]: namespacekaldi.html
[344]: classkaldi_1_1DiagGmm.html
[345]: namespacekaldi.html
[346]: structkaldi_1_1LbfgsOptions.html
[347]: namespacekaldi.html
[348]: classkaldi_1_1OnlineGenericBaseFeature.html
[349]: namespacekaldi.html
[350]: classkaldi_1_1SparseVector.html
[351]: namespacekaldi.html
[352]: classfst_1_1ArcticWeightTpl.html
[353]: namespacefst.html
[354]: classkaldi_1_1DiagGmmNormal.html
[355]: namespacekaldi.html
[356]: classkaldi_1_1LdaEstimate.html
[357]: namespacekaldi.html
[358]: structkaldi_1_1OnlineGmmAdaptationState.html
[359]: namespacekaldi.html
[360]: classkaldi_1_1differentiable__transform_1_1FmllrTransform_1_1SpeakerStats.html
[361]: namespacekaldi_1_1differentiable__transform.html
[362]: classkaldi_1_1ArpaFileParser.html
[363]: namespacekaldi.html
[364]: classkaldi_1_1differentiable__transform_1_1DifferentiableTransform.html
[365]: namespacekaldi_1_1differentiable__transform.html
[366]: structkaldi_1_1LdaEstimateOptions.html
[367]: namespacekaldi.html
[368]: structkaldi_1_1OnlineGmmDecodingAdaptationPolicyConfig.html
[369]: namespacekaldi.html
[370]: classkaldi_1_1differentiable__transform_1_1SpeakerStatsItf.html
[371]: namespacekaldi_1_1differentiable__transform.html
[372]: structkaldi_1_1ArpaLine.html
[373]: namespacekaldi.html
[374]: classkaldi_1_1discriminative_1_1DiscriminativeComputation.html
[375]: namespacekaldi_1_1discriminative.html
[376]: classkaldi_1_1nnet1_1_1LengthNormComponent.html
[377]: namespacekaldi_1_1nnet1.html
[378]: structkaldi_1_1OnlineGmmDecodingConfig.html
[379]: namespacekaldi.html
[380]: classkaldi_1_1nnet3_1_1SpecAugmentTimeMaskComponent.html
[381]: namespacekaldi_1_1nnet3.html
[382]: classkaldi_1_1ArpaLmCompiler.html
[383]: namespacekaldi.html
[384]: classkaldi_1_1nnet3_1_1DiscriminativeExampleMerger.html
[385]: namespacekaldi_1_1nnet3.html
[386]: structkaldi_1_1DecodableAmDiagGmmUnmapped_1_1LikelihoodCacheRecord.html
[387]: namespacekaldi.html
[388]: classkaldi_1_1OnlineGmmDecodingModels.html
[389]: namespacekaldi.html
[390]: classkaldi_1_1nnet3_1_1SpecAugmentTimeMaskComponentPrecomputedIndexes.html
[391]: namespacekaldi_1_1nnet3.html
[392]: classkaldi_1_1ArpaLmCompilerImpl.html
[393]: namespacekaldi.html
[394]: classkaldi_1_1nnet2_1_1DiscriminativeExampleSplitter.html
[395]: namespacekaldi_1_1nnet2.html
[396]: classkaldi_1_1nnet2_1_1LimitRankClass.html
[397]: namespacekaldi_1_1nnet2.html
[398]: classkaldi_1_1OnlineIvectorEstimationStats.html
[399]: namespacekaldi.html
[400]: classkaldi_1_1SpectrogramComputer.html
[401]: namespacekaldi.html
[402]: classkaldi_1_1ArpaLmCompilerImplInterface.html
[403]: namespacekaldi.html
[404]: classkaldi_1_1nnet2_1_1DiscriminativeExamplesRepository.html
[405]: namespacekaldi_1_1nnet2.html
[406]: structkaldi_1_1LinearCgdOptions.html
[407]: namespacekaldi.html
[408]: structkaldi_1_1OnlineIvectorExtractionConfig.html
[409]: namespacekaldi.html
[410]: structkaldi_1_1SpectrogramOptions.html
[411]: namespacekaldi.html
[412]: structkaldi_1_1ArpaParseOptions.html
[413]: namespacekaldi.html
[414]: structkaldi_1_1nnet2_1_1DiscriminativeNnetExample.html
[415]: namespacekaldi_1_1nnet2.html
[416]: classkaldi_1_1nnet3_1_1LinearComponent.html
[417]: namespacekaldi_1_1nnet3.html
[418]: structkaldi_1_1OnlineIvectorExtractionInfo.html
[419]: namespacekaldi.html
[420]: structkaldi_1_1SpeexOptions.html
[421]: namespacekaldi.html
[422]: classkaldi_1_1nnet1_1_1AveragePoolingComponent.html
[423]: namespacekaldi_1_1nnet1.html
[424]: structkaldi_1_1nnet3_1_1DiscriminativeObjectiveFunctionInfo.html
[425]: namespacekaldi_1_1nnet3.html
[426]: classkaldi_1_1LinearResample.html
[427]: namespacekaldi.html
[428]: structkaldi_1_1OnlineIvectorExtractorAdaptationState.html
[429]: namespacekaldi.html
[430]: classkaldi_1_1SphinxMatrixHolder.html
[431]: namespacekaldi.html
[432]: structkaldi_1_1discriminative_1_1DiscriminativeObjectiveInfo.html
[433]: namespacekaldi_1_1discriminative.html
[434]: classkaldi_1_1nnet1_1_1LinearTransform.html
[435]: namespacekaldi_1_1nnet1.html
[436]: classkaldi_1_1OnlineIvectorFeature.html
[437]: namespacekaldi.html
[438]: classkaldi_1_1nnet1_1_1Splice.html
[439]: namespacekaldi_1_1nnet1.html
[440]: structkaldi_1_1discriminative_1_1DiscriminativeOptions.html
[441]: namespacekaldi_1_1discriminative.html
[442]: classkaldi_1_1LinearVtln.html
[443]: namespacekaldi.html
[444]: classkaldi_1_1OnlineLdaInput.html
[445]: namespacekaldi.html
[446]: classkaldi_1_1nnet2_1_1SpliceComponent.html
[447]: namespacekaldi_1_1nnet2.html
[448]: classfst_1_1BackoffDeterministicOnDemandFst.html
[449]: namespacefst.html
[450]: structkaldi_1_1discriminative_1_1DiscriminativeSupervision.html
[451]: namespacekaldi_1_1discriminative.html
[452]: classfst_1_1LmExampleDeterministicOnDemandFst.html
[453]: namespacefst.html
[454]: classkaldi_1_1OnlineMatrixFeature.html
[455]: namespacekaldi.html
[456]: classkaldi_1_1nnet2_1_1SpliceMaxComponent.html
[457]: namespacekaldi_1_1nnet2.html
[458]: structkaldi_1_1decoder_1_1BackpointerToken.html
[459]: namespacekaldi_1_1decoder.html
[460]: classkaldi_1_1discriminative_1_1DiscriminativeSupervisionSplitter.html
[461]: namespacekaldi_1_1discriminative.html
[462]: classkaldi_1_1LmState.html
[463]: namespacekaldi.html
[464]: classkaldi_1_1OnlineMatrixInput.html
[465]: namespacekaldi.html
[466]: structkaldi_1_1nnet2_1_1SplitDiscriminativeExampleConfig.html
[467]: namespacekaldi_1_1nnet2.html
[468]: classkaldi_1_1nnet3_1_1BackpropTruncationComponent.html
[469]: namespacekaldi_1_1nnet3.html
[470]: classkaldi_1_1nnet2_1_1DiscTrainParallelClass.html
[471]: namespacekaldi_1_1nnet2.html
[472]: structkaldi_1_1MessageLogger_1_1Log.html
[473]: namespacekaldi.html
[474]: classkaldi_1_1nnet3_1_1OnlineNaturalGradient.html
[475]: namespacekaldi_1_1nnet3.html
[476]: structkaldi_1_1discriminative_1_1SplitDiscriminativeSupervisionOptions.html
[477]: namespacekaldi_1_1discriminative.html
[478]: classkaldi_1_1nnet3_1_1BackpropTruncationComponentPrecomputedIndexes.html
[479]: namespacekaldi_1_1nnet3.html
[480]: classkaldi_1_1nnet3_1_1DistributeComponent.html
[481]: namespacekaldi_1_1nnet3.html
[482]: structkaldi_1_1MessageLogger_1_1LogAndThrow.html
[483]: namespacekaldi.html
[484]: classkaldi_1_1nnet3_1_1OnlineNaturalGradientSimple.html
[485]: namespacekaldi_1_1nnet3.html
[486]: classkaldi_1_1SplitEventMap.html
[487]: namespacekaldi.html
[488]: classkaldi_1_1basic__filebuf.html
[489]: namespacekaldi.html
[490]: classkaldi_1_1nnet3_1_1DistributeComponentPrecomputedIndexes.html
[491]: namespacekaldi_1_1nnet3.html
[492]: classkaldi_1_1LogisticRegression.html
[493]: namespacekaldi.html
[494]: structkaldi_1_1OnlineNnet2DecodingConfig.html
[495]: namespacekaldi.html
[496]: structkaldi_1_1nnet2_1_1SplitExampleStats.html
[497]: namespacekaldi_1_1nnet2.html
[498]: classkaldi_1_1basic__pipebuf.html
[499]: namespacekaldi.html
[500]: classkaldi_1_1nnet2_1_1DoBackpropParallelClass.html
[501]: namespacekaldi_1_1nnet2.html
[502]: structkaldi_1_1LogisticRegressionConfig.html
[503]: namespacekaldi.html
[504]: structkaldi_1_1OnlineNnet2DecodingThreadedConfig.html
[505]: namespacekaldi.html
[506]: classkaldi_1_1SplitRadixComplexFft.html
[507]: namespacekaldi.html
[508]: classkaldi_1_1BasicHolder.html
[509]: namespacekaldi.html
[510]: structkaldi_1_1ParseOptions_1_1DocInfo.html
[511]: namespacekaldi.html
[512]: structkaldi_1_1LogMessageEnvelope.html
[513]: namespacekaldi.html
[514]: classkaldi_1_1OnlineNnet2FeaturePipeline.html
[515]: namespacekaldi.html
[516]: classkaldi_1_1SplitRadixRealFft.html
[517]: namespacekaldi.html
[518]: classkaldi_1_1BasicPairVectorHolder.html
[519]: namespacekaldi.html
[520]: classkaldi_1_1nnet1_1_1Dropout.html
[521]: namespacekaldi_1_1nnet1.html
[522]: classkaldi_1_1nnet2_1_1LogSoftmaxComponent.html
[523]: namespacekaldi_1_1nnet2.html
[524]: structkaldi_1_1OnlineNnet2FeaturePipelineConfig.html
[525]: namespacekaldi.html
[526]: classkaldi_1_1SpMatrix.html
[527]: namespacekaldi.html
[528]: classkaldi_1_1BasicVectorHolder.html
[529]: namespacekaldi.html
[530]: classkaldi_1_1nnet2_1_1DropoutComponent.html
[531]: namespacekaldi_1_1nnet2.html
[532]: classkaldi_1_1nnet3_1_1LogSoftmaxComponent.html
[533]: namespacekaldi_1_1nnet3.html
[534]: structkaldi_1_1OnlineNnet2FeaturePipelineInfo.html
[535]: namespacekaldi.html
[536]: classkaldi_1_1StandardInputImpl.html
[537]: namespacekaldi.html
[538]: classkaldi_1_1BasicVectorVectorHolder.html
[539]: namespacekaldi.html
[540]: classkaldi_1_1nnet3_1_1DropoutComponent.html
[541]: namespacekaldi_1_1nnet3.html
[542]: classkaldi_1_1nnet1_1_1LossItf.html
[543]: namespacekaldi_1_1nnet1.html
[544]: classkaldi_1_1OnlinePaSource.html
[545]: namespacekaldi.html
[546]: classkaldi_1_1StandardOutputImpl.html
[547]: namespacekaldi.html
[548]: classkaldi_1_1BasisFmllrAccus.html
[549]: namespacekaldi.html
[550]: classkaldi_1_1nnet3_1_1DropoutMaskComponent.html
[551]: namespacekaldi_1_1nnet3.html
[552]: structkaldi_1_1nnet1_1_1LossOptions.html
[553]: namespacekaldi_1_1nnet1.html
[554]: classkaldi_1_1OnlinePitchFeature.html
[555]: namespacekaldi.html
[556]: structkaldi_1_1PitchFrameInfo_1_1StateInfo.html
[557]: namespacekaldi.html
[558]: classkaldi_1_1BasisFmllrEstimate.html
[559]: namespacekaldi.html
[560]: structkaldi_1_1DummyOptions.html
[561]: namespacekaldi.html
[562]: classkaldi_1_1nnet3_1_1LstmNonlinearityComponent.html
[563]: namespacekaldi_1_1nnet3.html
[564]: classkaldi_1_1OnlinePitchFeatureImpl.html
[565]: namespacekaldi.html
[566]: classfst_1_1StateIterator_3_01TrivialFactorWeightFst_3_01A_00_01F_01_4_01_4.html
[567]: namespacefst.html
[568]: structkaldi_1_1BasisFmllrOptions.html
[569]: namespacekaldi.html
[570]: classkaldi_1_1nnet1_1_1LstmProjected.html
[571]: namespacekaldi_1_1nnet1.html
[572]: classkaldi_1_1nnet2_1_1OnlinePreconditioner.html
[573]: namespacekaldi_1_1nnet2.html
[574]: classkaldi_1_1nnet3_1_1StatisticsExtractionComponent.html
[575]: namespacekaldi_1_1nnet3.html
[576]: classkaldi_1_1nnet3_1_1BatchedXvectorComputer.html
[577]: namespacekaldi_1_1nnet3.html
[578]: classkaldi_1_1nnet2_1_1OnlinePreconditionerSimple.html
[579]: namespacekaldi_1_1nnet2.html
[580]: classkaldi_1_1nnet3_1_1StatisticsExtractionComponentPrecomputedIndexes.html
[581]: namespacekaldi_1_1nnet3.html
[582]: structkaldi_1_1nnet3_1_1BatchedXvectorComputerOptions.html
[583]: namespacekaldi_1_1nnet3.html
[584]: structkaldi_1_1EbwAmSgmm2Options.html
[585]: namespacekaldi.html
[586]: classkaldi_1_1OnlineProcessPitch.html
[587]: namespacekaldi.html
[588]: classkaldi_1_1nnet3_1_1StatisticsPoolingComponent.html
[589]: namespacekaldi_1_1nnet3.html
[590]: classkaldi_1_1nnet3_1_1BatchNormComponent.html
[591]: namespacekaldi_1_1nnet3.html
[592]: classkaldi_1_1EbwAmSgmm2Updater.html
[593]: namespacekaldi.html
[594]: structkaldi_1_1MapDiagGmmOptions.html
[595]: namespacekaldi.html
[596]: classkaldi_1_1OnlineSilenceWeighting.html
[597]: namespacekaldi.html
[598]: classkaldi_1_1nnet3_1_1StatisticsPoolingComponentPrecomputedIndexes.html
[599]: namespacekaldi_1_1nnet3.html
[600]: structkaldi_1_1LatticeFasterOnlineDecoderTpl_1_1BestPathIterator.html
[601]: namespacekaldi.html
[602]: classEbwAmSgmmUpdater.html
[603]: classfst_1_1MapInputSymbolsMapper.html
[604]: namespacefst.html
[605]: structkaldi_1_1OnlineSilenceWeightingConfig.html
[606]: namespacekaldi.html
[607]: structkaldi_1_1nnet2_1_1NnetStats_1_1StatsElement.html
[608]: namespacekaldi_1_1nnet2.html
[609]: structkaldi_1_1LatticeIncrementalOnlineDecoderTpl_1_1BestPathIterator.html
[610]: namespacekaldi.html
[611]: structkaldi_1_1EbwOptions.html
[612]: namespacekaldi.html
[613]: structkaldi_1_1MapTransitionUpdateConfig.html
[614]: namespacekaldi.html
[615]: classkaldi_1_1OnlineSpeexDecoder.html
[616]: namespacekaldi.html
[617]: structkaldi_1_1nnet3_1_1ExampleMergingStats_1_1StatsForExampleSize.html
[618]: namespacekaldi_1_1nnet3.html
[619]: classkaldi_1_1BiglmFasterDecoder.html
[620]: namespacekaldi.html
[621]: classkaldi_1_1EbwUpdatePhoneVectorsClass.html
[622]: namespacekaldi.html
[623]: classkaldi_1_1Matrix.html
[624]: namespacekaldi.html
[625]: classkaldi_1_1OnlineSpeexEncoder.html
[626]: namespacekaldi.html
[627]: structkaldi_1_1decoder_1_1StdToken.html
[628]: namespacekaldi_1_1decoder.html
[629]: structkaldi_1_1BiglmFasterDecoderOptions.html
[630]: namespacekaldi.html
[631]: structkaldi_1_1EbwWeightOptions.html
[632]: namespacekaldi.html
[633]: structkaldi_1_1nnet3_1_1MatrixAccesses.html
[634]: namespacekaldi_1_1nnet3.html
[635]: classkaldi_1_1OnlineSpliceFrames.html
[636]: namespacekaldi.html
[637]: classfst_1_1StdToLatticeMapper.html
[638]: namespacefst.html
[639]: classkaldi_1_1nnet3_1_1BinarySumDescriptor.html
[640]: namespacekaldi_1_1nnet3.html
[641]: classkaldi_1_1EigenvalueDecomposition.html
[642]: namespacekaldi.html
[643]: classkaldi_1_1MatrixBase.html
[644]: namespacekaldi.html
[645]: structkaldi_1_1OnlineSpliceOptions.html
[646]: namespacekaldi.html
[647]: classkaldi_1_1nnet1_1_1StdVectorRandomizer.html
[648]: namespacekaldi_1_1nnet1.html
[649]: classkaldi_1_1nnet2_1_1BlockAffineComponent.html
[650]: namespacekaldi_1_1nnet2.html
[651]: structkaldi_1_1HashList_1_1Elem.html
[652]: namespacekaldi.html
[653]: classkaldi_1_1nnet1_1_1MatrixBuffer.html
[654]: namespacekaldi_1_1nnet1.html
[655]: classkaldi_1_1OnlineTcpVectorSource.html
[656]: namespacekaldi.html
[657]: structkaldi_1_1nnet3_1_1Compiler_1_1StepInfo.html
[658]: namespacekaldi_1_1nnet3.html
[659]: classkaldi_1_1nnet3_1_1BlockAffineComponent.html
[660]: namespacekaldi_1_1nnet3.html
[661]: structfst_1_1LatticeDeterminizer_1_1Element.html
[662]: namespacefst.html
[663]: structkaldi_1_1nnet1_1_1MatrixBufferOptions.html
[664]: namespacekaldi_1_1nnet1.html
[665]: classkaldi_1_1OnlineTimer.html
[666]: namespacekaldi.html
[667]: structkaldi_1_1StringHasher.html
[668]: namespacekaldi.html
[669]: classkaldi_1_1nnet2_1_1BlockAffineComponentPreconditioned.html
[670]: namespacekaldi_1_1nnet2.html
[671]: structfst_1_1DeterminizerStar_1_1Element.html
[672]: namespacefst.html
[673]: structkaldi_1_1nnet3_1_1MemoryCompressionOptimizer_1_1MatrixCompressInfo.html
[674]: namespacekaldi_1_1nnet3.html
[675]: classkaldi_1_1OnlineTimingStats.html
[676]: namespacekaldi.html
[677]: classfst_1_1StringRepository.html
[678]: namespacefst.html
[679]: classkaldi_1_1nnet3_1_1BlockFactorizedTdnnComponent.html
[680]: namespacekaldi_1_1nnet3.html
[681]: structfst_1_1internal_1_1TrivialFactorWeightFstImpl_1_1Element.html
[682]: namespacefst_1_1internal.html
[683]: structkaldi_1_1nnet3_1_1NnetComputation_1_1MatrixDebugInfo.html
[684]: namespacekaldi_1_1nnet3.html
[685]: classkaldi_1_1OnlineTransform.html
[686]: namespacekaldi.html
[687]: classkaldi_1_1SubMatrix.html
[688]: namespacekaldi.html
[689]: structkaldi_1_1CuBlockMatrix_1_1BlockMatrixData.html
[690]: namespacekaldi.html
[691]: structfst_1_1LatticeDeterminizerPruned_1_1Element.html
[692]: namespacefst.html
[693]: structMatrixDim__.html
[694]: classkaldi_1_1OnlineUdpInput.html
[695]: namespacekaldi.html
[696]: structkaldi_1_1nnet3_1_1ComputationRenumberer_1_1SubMatrixHasher.html
[697]: namespacekaldi_1_1nnet3.html
[698]: classkaldi_1_1nnet1_1_1BlockSoftmax.html
[699]: namespacekaldi_1_1nnet1.html
[700]: classfst_1_1internal_1_1TrivialFactorWeightFstImpl_1_1ElementEqual.html
[701]: namespacefst_1_1internal.html
[702]: structMatrixElement.html
[703]: classkaldi_1_1OnlineVectorSource.html
[704]: namespacekaldi.html
[705]: structkaldi_1_1nnet3_1_1NnetComputation_1_1SubMatrixInfo.html
[706]: namespacekaldi_1_1nnet3.html
[707]: classkaldi_1_1nnet1_1_1BlstmProjected.html
[708]: namespacekaldi_1_1nnet1.html
[709]: classfst_1_1internal_1_1TrivialFactorWeightFstImpl_1_1ElementKey.html
[710]: namespacefst_1_1internal.html
[711]: classkaldi_1_1nnet3_1_1MatrixExtender.html
[712]: namespacekaldi_1_1nnet3.html
[713]: classkaldi_1_1OptimizableInterface.html
[714]: namespacekaldi.html
[715]: classfst_1_1DeterminizerStar_1_1SubsetEqual.html
[716]: namespacefst.html
[717]: classkaldi_1_1BottomUpClusterer.html
[718]: namespacekaldi.html
[719]: classkaldi_1_1nnet3_1_1ElementwiseProductComponent.html
[720]: namespacekaldi_1_1nnet3.html
[721]: structkaldi_1_1nnet3_1_1NnetComputation_1_1MatrixInfo.html
[722]: namespacekaldi_1_1nnet3.html
[723]: classkaldi_1_1OptimizeLbfgs.html
[724]: namespacekaldi.html
[725]: classfst_1_1LatticeDeterminizerPruned_1_1SubsetEqual.html
[726]: namespacefst.html
[727]: structfst_1_1LatticeStringRepository_1_1Entry.html
[728]: namespacefst.html
[729]: structkaldi_1_1nnet3_1_1DerivativeTimeLimiter_1_1MatrixPruneInfo.html
[730]: namespacekaldi_1_1nnet3.html
[731]: classkaldi_1_1nnet3_1_1OptionalSumDescriptor.html
[732]: namespacekaldi_1_1nnet3.html
[733]: classfst_1_1LatticeDeterminizer_1_1SubsetEqual.html
[734]: namespacefst.html
[735]: classfst_1_1LatticeStringRepository_1_1EntryEqual.html
[736]: namespacefst.html
[737]: classkaldi_1_1nnet1_1_1MatrixRandomizer.html
[738]: namespacekaldi_1_1nnet1.html
[739]: structkaldi_1_1SimpleOptions_1_1OptionInfo.html
[740]: namespacekaldi.html
[741]: classfst_1_1LatticeDeterminizer_1_1SubsetEqualStates.html
[742]: namespacefst.html
[743]: classCacheArcIterator.html
[744]: classfst_1_1LatticeStringRepository_1_1EntryKey.html
[745]: namespacefst.html
[746]: structkaldi_1_1nnet3_1_1MaxChangeStats.html
[747]: namespacekaldi_1_1nnet3.html
[748]: classkaldi_1_1OptionsItf.html
[749]: namespacekaldi.html
[750]: classfst_1_1DeterminizerStar_1_1SubsetEqualStates.html
[751]: namespacefst.html
[752]: classfst_1_1CacheDeterministicOnDemandFst.html
[753]: namespacefst.html
[754]: classfst_1_1DeterminizerStar_1_1EpsilonClosure.html
[755]: namespacefst.html
[756]: classkaldi_1_1nnet2_1_1MaxoutComponent.html
[757]: namespacekaldi_1_1nnet2.html
[758]: classkaldi_1_1OtherReal.html
[759]: namespacekaldi.html
[760]: classfst_1_1LatticeDeterminizerPruned_1_1SubsetEqualStates.html
[761]: namespacefst.html
[762]: classCacheImpl.html
[763]: structfst_1_1DeterminizerStar_1_1EpsilonClosure_1_1EpsilonClosureInfo.html
[764]: namespacefst.html
[765]: classkaldi_1_1nnet2_1_1MaxpoolingComponent.html
[766]: namespacekaldi_1_1nnet2.html
[767]: classkaldi_1_1OtherReal_3_01double_01_4.html
[768]: namespacekaldi.html
[769]: classfst_1_1DeterminizerStar_1_1SubsetKey.html
[770]: namespacefst.html
[771]: classCacheOptions.html
[772]: structfst_1_1CompactLatticeMinimizer_1_1EquivalenceSorter.html
[773]: namespacefst.html
[774]: classkaldi_1_1nnet3_1_1MaxpoolingComponent.html
[775]: namespacekaldi_1_1nnet3.html
[776]: classkaldi_1_1OtherReal_3_01float_01_4.html
[777]: namespacekaldi.html
[778]: classfst_1_1LatticeDeterminizer_1_1SubsetKey.html
[779]: namespacefst.html
[780]: classCacheStateIterator.html
[781]: structkaldi_1_1error__stats.html
[782]: namespacekaldi.html
[783]: classkaldi_1_1nnet1_1_1MaxPoolingComponent.html
[784]: namespacekaldi_1_1nnet1.html
[785]: classkaldi_1_1Output.html
[786]: namespacekaldi.html
[787]: classfst_1_1LatticeDeterminizerPruned_1_1SubsetKey.html
[788]: namespacefst.html
[789]: classkaldi_1_1nnet3_1_1CachingOptimizingCompiler.html
[790]: namespacekaldi_1_1nnet3.html
[791]: classkaldi_1_1EventMap.html
[792]: namespacekaldi.html
[793]: classkaldi_1_1MelBanks.html
[794]: namespacekaldi.html
[795]: classkaldi_1_1OutputImplBase.html
[796]: namespacekaldi.html
[797]: structkaldi_1_1Sgmm2LikelihoodCache_1_1SubstateCacheElement.html
[798]: namespacekaldi.html
[799]: structkaldi_1_1nnet3_1_1CachingOptimizingCompilerOptions.html
[800]: namespacekaldi_1_1nnet3.html
[801]: structkaldi_1_1EventMapVectorEqual.html
[802]: namespacekaldi.html
[803]: structkaldi_1_1MelBanksOptions.html
[804]: namespacekaldi.html
[805]: structfst_1_1LatticeDeterminizerPruned_1_1OutputState.html
[806]: namespacefst.html
[807]: classkaldi_1_1SubVector.html
[808]: namespacekaldi.html
[809]: classkaldi_1_1nnet3_1_1ChainExampleMerger.html
[810]: namespacekaldi_1_1nnet3.html
[811]: structkaldi_1_1EventMapVectorHash.html
[812]: namespacekaldi.html
[813]: structkaldi_1_1nnet3_1_1RestrictedAttentionComponent_1_1Memo.html
[814]: namespacekaldi_1_1nnet3.html
[815]: classkaldi_1_1nnet3_1_1SumBlockComponent.html
[816]: namespacekaldi_1_1nnet3.html
[817]: structkaldi_1_1nnet3_1_1ChainObjectiveInfo.html
[818]: namespacekaldi_1_1nnet3.html
[819]: classkaldi_1_1ExampleClass.html
[820]: namespacekaldi.html
[821]: structkaldi_1_1nnet3_1_1BatchNormComponent_1_1Memo.html
[822]: namespacekaldi_1_1nnet3.html
[823]: classkaldi_1_1nnet3_1_1SumDescriptor.html
[824]: namespacekaldi_1_1nnet3.html
[825]: structkaldi_1_1nnet3_1_1CheckComputationOptions.html
[826]: namespacekaldi_1_1nnet3.html
[827]: classkaldi_1_1ExampleFeatureComputer.html
[828]: namespacekaldi.html
[829]: classkaldi_1_1nnet3_1_1MemoryCompressionOptimizer.html
[830]: namespacekaldi_1_1nnet3.html
[831]: classkaldi_1_1PackedMatrix.html
[832]: namespacekaldi.html
[833]: classkaldi_1_1nnet2_1_1SumGroupComponent.html
[834]: namespacekaldi_1_1nnet2.html
[835]: structkaldi_1_1LmState_1_1ChildrenVectorLessThan.html
[836]: namespacekaldi.html
[837]: structkaldi_1_1ExampleFeatureComputerOptions.html
[838]: namespacekaldi.html
[839]: classkaldi_1_1MessageLogger.html
[840]: namespacekaldi.html
[841]: classfst_1_1LatticeDeterminizer_1_1PairComparator.html
[842]: namespacefst.html
[843]: classkaldi_1_1nnet3_1_1SumGroupComponent.html
[844]: namespacekaldi_1_1nnet3.html
[845]: unionkaldi_1_1LmState_1_1ChildType.html
[846]: namespacekaldi.html
[847]: structkaldi_1_1nnet3_1_1ExampleGenerationConfig.html
[848]: namespacekaldi_1_1nnet3.html
[849]: classkaldi_1_1MfccComputer.html
[850]: namespacekaldi.html
[851]: classfst_1_1DeterminizerStar_1_1PairComparator.html
[852]: namespacefst.html
[853]: classkaldi_1_1nnet3_1_1SvdApplier.html
[854]: namespacekaldi_1_1nnet3.html
[855]: classkaldi_1_1nnet2_1_1ChunkInfo.html
[856]: namespacekaldi_1_1nnet2.html
[857]: classkaldi_1_1nnet3_1_1ExampleMerger.html
[858]: namespacekaldi_1_1nnet3.html
[859]: structkaldi_1_1MfccOptions.html
[860]: namespacekaldi.html
[861]: classfst_1_1LatticeDeterminizerPruned_1_1PairComparator.html
[862]: namespacefst.html
[863]: classkaldi_1_1nnet3_1_1SwitchingForwardingDescriptor.html
[864]: namespacekaldi_1_1nnet3.html
[865]: structkaldi_1_1nnet3_1_1ChunkInfo.html
[866]: namespacekaldi_1_1nnet3.html
[867]: classkaldi_1_1nnet3_1_1ExampleMergingConfig.html
[868]: namespacekaldi_1_1nnet3.html
[869]: classkaldi_1_1differentiable__transform_1_1SimpleMeanTransform_1_1MinibatchInfo.html
[870]: namespacekaldi_1_1differentiable__transform.html
[871]: structkaldi_1_1RandomAccessTableReaderSortedArchiveImpl_1_1PairCompare.html
[872]: namespacekaldi.html
[873]: structrnnlm_1_1synapse.html
[874]: namespacernnlm.html
[875]: structkaldi_1_1nnet3_1_1ChunkTimeInfo.html
[876]: namespacekaldi_1_1nnet3.html
[877]: classkaldi_1_1nnet3_1_1ExampleMergingStats.html
[878]: namespacekaldi_1_1nnet3.html
[879]: classkaldi_1_1differentiable__transform_1_1FmllrTransform_1_1MinibatchInfo.html
[880]: namespacekaldi_1_1differentiable__transform.html
[881]: structkaldi_1_1PairHasher.html
[882]: namespacekaldi.html
[883]: structkaldi_1_1nnet3_1_1CindexHasher.html
[884]: namespacekaldi_1_1nnet3.html
[885]: classkaldi_1_1nnet2_1_1ExamplesRepository.html
[886]: namespacekaldi_1_1nnet2.html
[887]: classkaldi_1_1differentiable__transform_1_1MinibatchInfoItf.html
[888]: namespacekaldi_1_1differentiable__transform.html
[889]: structkaldi_1_1nnet3_1_1PairIsEqualComparator.html
[890]: namespacekaldi_1_1nnet3.html
[891]: structkaldi_1_1nnet3_1_1ComputationGraphBuilder_1_1CindexInfo.html
[892]: namespacekaldi_1_1nnet3.html
[893]: structfst_1_1GrammarFst_1_1ExpandedState.html
[894]: namespacefst.html
[895]: structkaldi_1_1nnet3_1_1NnetBatchComputer_1_1MinibatchSizeInfo.html
[896]: namespacekaldi_1_1nnet3.html
[897]: classkaldi_1_1nnet1_1_1ParallelComponent.html
[898]: namespacekaldi_1_1nnet1.html
[899]: structfst_1_1TableComposeCache.html
[900]: namespacefst.html
[901]: classkaldi_1_1nnet3_1_1CindexSet.html
[902]: namespacekaldi_1_1nnet3.html
[903]: classkaldi_1_1MinimumBayesRisk.html
[904]: namespacekaldi.html
[905]: classkaldi_1_1nnet1_1_1ParametricRelu.html
[906]: namespacekaldi_1_1nnet1.html
[907]: structfst_1_1TableComposeOptions.html
[908]: namespacefst.html
[909]: structkaldi_1_1nnet3_1_1CindexVectorHasher.html
[910]: namespacekaldi_1_1nnet3.html
[911]: structkaldi_1_1MinimumBayesRiskOptions.html
[912]: namespacekaldi.html
[913]: classkaldi_1_1ParseOptions.html
[914]: namespacekaldi.html
[915]: classkaldi_1_1TableEventMap.html
[916]: namespacekaldi.html
[917]: structkaldi_1_1PldaStats_1_1ClassInfo.html
[918]: namespacekaldi.html
[919]: classkaldi_1_1FasterDecoder.html
[920]: namespacekaldi.html
[921]: structkaldi_1_1nnet3_1_1MiscComputationInfo.html
[922]: namespacekaldi_1_1nnet3.html
[923]: structkaldi_1_1Sgmm2LikelihoodCache_1_1PdfCacheElement.html
[924]: namespacekaldi.html
[925]: classfst_1_1TableMatcher.html
[926]: namespacefst.html
[927]: structkaldi_1_1ClatRescoreTuple.html
[928]: namespacekaldi.html
[929]: structkaldi_1_1FasterDecoderOptions.html
[930]: namespacekaldi.html
[931]: classkaldi_1_1MleAmSgmm2Accs.html
[932]: namespacekaldi.html
[933]: classkaldi_1_1nnet1_1_1PdfPrior.html
[934]: namespacekaldi_1_1nnet1.html
[935]: classfst_1_1TableMatcherImpl.html
[936]: namespacefst.html
[937]: classkaldi_1_1nnet3_1_1ClipGradientComponent.html
[938]: namespacekaldi_1_1nnet3.html
[939]: classkaldi_1_1nnet2_1_1FastNnetCombiner.html
[940]: namespacekaldi_1_1nnet2.html
[941]: structkaldi_1_1MleAmSgmm2Options.html
[942]: namespacekaldi.html
[943]: structkaldi_1_1nnet1_1_1PdfPriorOptions.html
[944]: namespacekaldi_1_1nnet1.html
[945]: structfst_1_1TableMatcherOptions.html
[946]: namespacefst.html
[947]: classkaldi_1_1Clusterable.html
[948]: namespacekaldi.html
[949]: classkaldi_1_1FbankComputer.html
[950]: namespacekaldi.html
[951]: classkaldi_1_1MleAmSgmm2Updater.html
[952]: namespacekaldi.html
[953]: structkaldi_1_1CompressedMatrix_1_1PerColHeader.html
[954]: namespacekaldi.html
[955]: classkaldi_1_1TableWriter.html
[956]: namespacekaldi.html
[957]: structkaldi_1_1ClusterKMeansOptions.html
[958]: namespacekaldi.html
[959]: structkaldi_1_1FbankOptions.html
[960]: namespacekaldi.html
[961]: classMleAmSgmmUpdater.html
[962]: structkaldi_1_1nnet3_1_1PerDimObjectiveInfo.html
[963]: namespacekaldi_1_1nnet3.html
[964]: classkaldi_1_1TableWriterArchiveImpl.html
[965]: namespacekaldi.html
[966]: structkaldi_1_1nnet3_1_1CollapseModelConfig.html
[967]: namespacekaldi_1_1nnet3.html
[968]: classkaldi_1_1FeatureTransformEstimate.html
[969]: namespacekaldi.html
[970]: structkaldi_1_1MleDiagGmmOptions.html
[971]: namespacekaldi.html
[972]: classkaldi_1_1nnet3_1_1PerElementOffsetComponent.html
[973]: namespacekaldi_1_1nnet3.html
[974]: classkaldi_1_1TableWriterBothImpl.html
[975]: namespacekaldi.html
[976]: structkaldi_1_1nnet3_1_1NnetComputation_1_1Command.html
[977]: namespacekaldi_1_1nnet3.html
[978]: classkaldi_1_1FeatureTransformEstimateMulti.html
[979]: namespacekaldi.html
[980]: structkaldi_1_1MleFullGmmOptions.html
[981]: namespacekaldi.html
[982]: classkaldi_1_1nnet3_1_1PerElementScaleComponent.html
[983]: namespacekaldi_1_1nnet3.html
[984]: classkaldi_1_1TableWriterImplBase.html
[985]: namespacekaldi.html
[986]: structkaldi_1_1nnet3_1_1CommandAttributes.html
[987]: namespacekaldi_1_1nnet3.html
[988]: structkaldi_1_1FeatureTransformEstimateOptions.html
[989]: namespacekaldi.html
[990]: classkaldi_1_1MleSgmm2SpeakerAccs.html
[991]: namespacekaldi.html
[992]: classkaldi_1_1nnet2_1_1PermuteComponent.html
[993]: namespacekaldi_1_1nnet2.html
[994]: classkaldi_1_1TableWriterScriptImpl.html
[995]: namespacekaldi.html
[996]: structkaldi_1_1nnet3_1_1NnetComputer_1_1CommandDebugInfo.html
[997]: namespacekaldi_1_1nnet3.html
[998]: structkaldi_1_1FeatureWindowFunction.html
[999]: namespacekaldi.html
[1000]: structkaldi_1_1MleTransitionUpdateConfig.html
[1001]: namespacekaldi.html
[1002]: classkaldi_1_1nnet3_1_1PermuteComponent.html
[1003]: namespacekaldi_1_1nnet3.html
[1004]: classkaldi_1_1nnet1_1_1Tanh.html
[1005]: namespacekaldi_1_1nnet1.html
[1006]: structkaldi_1_1nnet3_1_1CommandPairComparator.html
[1007]: namespacekaldi_1_1nnet3.html
[1008]: classkaldi_1_1FileInputImpl.html
[1009]: namespacekaldi.html
[1010]: classkaldi_1_1MlltAccs.html
[1011]: namespacekaldi.html
[1012]: structkaldi_1_1PhoneAlignLatticeOptions.html
[1013]: namespacekaldi.html
[1014]: classkaldi_1_1nnet3_1_1TanhComponent.html
[1015]: namespacekaldi_1_1nnet3.html
[1016]: classkaldi_1_1CompactLatticeHolder.html
[1017]: namespacekaldi.html
[1018]: classkaldi_1_1FileOutputImpl.html
[1019]: namespacekaldi.html
[1020]: classkaldi_1_1nnet3_1_1ModelCollapser.html
[1021]: namespacekaldi_1_1nnet3.html
[1022]: classkaldi_1_1PipeInputImpl.html
[1023]: namespacekaldi.html
[1024]: classkaldi_1_1nnet2_1_1TanhComponent.html
[1025]: namespacekaldi_1_1nnet2.html
[1026]: classfst_1_1CompactLatticeMinimizer.html
[1027]: namespacefst.html
[1028]: classkaldi_1_1nnet2_1_1FisherComputationClass.html
[1029]: namespacekaldi_1_1nnet2.html
[1030]: classkaldi_1_1nnet3_1_1ModelUpdateConsolidator.html
[1031]: namespacekaldi_1_1nnet3.html
[1032]: classkaldi_1_1PipeOutputImpl.html
[1033]: namespacekaldi.html
[1034]: structkaldi_1_1nnet3_1_1TarjanNode.html
[1035]: namespacekaldi_1_1nnet3.html
[1036]: classfst_1_1CompactLatticePusher.html
[1037]: namespacefst.html
[1038]: classkaldi_1_1nnet2_1_1FixedAffineComponent.html
[1039]: namespacekaldi_1_1nnet2.html
[1040]: structkaldi_1_1nnet3_1_1SvdApplier_1_1ModifiedComponentInfo.html
[1041]: namespacekaldi_1_1nnet3.html
[1042]: structkaldi_1_1PitchExtractionOptions.html
[1043]: namespacekaldi.html
[1044]: structfst_1_1PruneSpecialClass_1_1Task.html
[1045]: namespacefst.html
[1046]: classkaldi_1_1CompactLatticeToKwsProductFstMapper.html
[1047]: namespacekaldi.html
[1048]: classkaldi_1_1nnet3_1_1FixedAffineComponent.html
[1049]: namespacekaldi_1_1nnet3.html
[1050]: classkaldi_1_1nnet1_1_1Mse.html
[1051]: namespacekaldi_1_1nnet1.html
[1052]: classkaldi_1_1PitchFrameInfo.html
[1053]: namespacekaldi.html
[1054]: structfst_1_1LatticeDeterminizerPruned_1_1Task.html
[1055]: namespacefst.html
[1056]: classfst_1_1CompactLatticeWeightCommonDivisorTpl.html
[1057]: namespacefst.html
[1058]: classkaldi_1_1nnet2_1_1FixedBiasComponent.html
[1059]: namespacekaldi_1_1nnet2.html
[1060]: classkaldi_1_1nnet1_1_1MultiBasisComponent.html
[1061]: namespacekaldi_1_1nnet1.html
[1062]: classkaldi_1_1PitchInterpolator.html
[1063]: namespacekaldi.html
[1064]: structfst_1_1LatticeDeterminizerPruned_1_1TaskCompare.html
[1065]: namespacefst.html
[1066]: classfst_1_1CompactLatticeWeightTpl.html
[1067]: namespacefst.html
[1068]: classkaldi_1_1nnet3_1_1FixedBiasComponent.html
[1069]: namespacekaldi_1_1nnet3.html
[1070]: structkaldi_1_1nnet3_1_1RowOpsSplitter_1_1MultiIndexSplitInfo.html
[1071]: namespacekaldi_1_1nnet3.html
[1072]: structkaldi_1_1PitchInterpolatorOptions.html
[1073]: namespacekaldi.html
[1074]: classkaldi_1_1TaskSequencer.html
[1075]: namespacekaldi.html
[1076]: structkaldi_1_1sparse__vector__utils_1_1CompareFirst.html
[1077]: namespacekaldi_1_1sparse__vector__utils.html
[1078]: classkaldi_1_1nnet2_1_1FixedLinearComponent.html
[1079]: namespacekaldi_1_1nnet2.html
[1080]: classkaldi_1_1nnet1_1_1MultistreamComponent.html
[1081]: namespacekaldi_1_1nnet1.html
[1082]: structkaldi_1_1PitchInterpolatorStats.html
[1083]: namespacekaldi.html
[1084]: structkaldi_1_1TaskSequencerConfig.html
[1085]: namespacekaldi.html
[1086]: structkaldi_1_1CompareFirstMemberOfPair.html
[1087]: namespacekaldi.html
[1088]: classkaldi_1_1nnet2_1_1FixedScaleComponent.html
[1089]: namespacekaldi_1_1nnet2.html
[1090]: classkaldi_1_1nnet1_1_1MultiTaskLoss.html
[1091]: namespacekaldi_1_1nnet1.html
[1092]: classkaldi_1_1Plda.html
[1093]: namespacekaldi.html
[1094]: classkaldi_1_1TcpServer.html
[1095]: namespacekaldi.html
[1096]: structkaldi_1_1nnet3_1_1ComparePair.html
[1097]: namespacekaldi_1_1nnet3.html
[1098]: classkaldi_1_1nnet3_1_1FixedScaleComponent.html
[1099]: namespacekaldi_1_1nnet3.html
[1100]: classkaldi_1_1MultiThreadable.html
[1101]: namespacekaldi.html
[1102]: structkaldi_1_1PldaConfig.html
[1103]: namespacekaldi.html
[1104]: classkaldi_1_1nnet3_1_1TdnnComponent.html
[1105]: namespacekaldi_1_1nnet3.html
[1106]: structkaldi_1_1ComparePosteriorByPdfs.html
[1107]: namespacekaldi.html
[1108]: classFloatWeightTpl.html
[1109]: classkaldi_1_1MultiThreader.html
[1110]: namespacekaldi.html
[1111]: structkaldi_1_1PldaEstimationConfig.html
[1112]: namespacekaldi.html
[1113]: structfst_1_1LatticeDeterminizer_1_1TempArc.html
[1114]: namespacefst.html
[1115]: structkaldi_1_1CompareReverseSecond.html
[1116]: namespacekaldi.html
[1117]: classkaldi_1_1FmllrDiagGmmAccs.html
[1118]: namespacekaldi.html
[1119]: classkaldi_1_1MyTaskClass.html
[1120]: namespacekaldi.html
[1121]: classkaldi_1_1PldaEstimator.html
[1122]: namespacekaldi.html
[1123]: structfst_1_1LatticeDeterminizerPruned_1_1TempArc.html
[1124]: namespacefst.html
[1125]: classkaldi_1_1CompartmentalizedBottomUpClusterer.html
[1126]: namespacekaldi.html
[1127]: structkaldi_1_1FmllrOptions.html
[1128]: namespacekaldi.html
[1129]: classkaldi_1_1MyThreadClass.html
[1130]: namespacekaldi.html
[1131]: classkaldi_1_1PldaStats.html
[1132]: namespacekaldi.html
[1133]: structfst_1_1DeterminizerStar_1_1TempArc.html
[1134]: namespacefst.html
[1135]: structkaldi_1_1CompBotClustElem.html
[1136]: namespacekaldi.html
[1137]: classkaldi_1_1FmllrRawAccs.html
[1138]: namespacekaldi.html
[1139]: classkaldi_1_1PldaUnsupervisedAdaptor.html
[1140]: namespacekaldi.html
[1141]: classTestFunction.html
[1142]: classkaldi_1_1nnet3_1_1Compiler.html
[1143]: namespacekaldi_1_1nnet3.html
[1144]: structkaldi_1_1FmllrRawOptions.html
[1145]: namespacekaldi.html
[1146]: structkaldi_1_1PldaUnsupervisedAdaptorConfig.html
[1147]: namespacekaldi.html
[1148]: structfst_1_1TestFunctor.html
[1149]: namespacefst.html
[1150]: structkaldi_1_1nnet3_1_1CompilerOptions.html
[1151]: namespacekaldi_1_1nnet3.html
[1152]: classkaldi_1_1FmllrSgmm2Accs.html
[1153]: namespacekaldi.html
[1154]: classkaldi_1_1nnet3_1_1NaturalGradientAffineComponent.html
[1155]: namespacekaldi_1_1nnet3.html
[1156]: classkaldi_1_1PlpComputer.html
[1157]: namespacekaldi.html
[1158]: classkaldi_1_1ThreadSynchronizer.html
[1159]: namespacekaldi.html
[1160]: classkaldi_1_1nnet2_1_1Component.html
[1161]: namespacekaldi_1_1nnet2.html
[1162]: classkaldi_1_1differentiable__transform_1_1FmllrTransform.html
[1163]: namespacekaldi_1_1differentiable__transform.html
[1164]: classkaldi_1_1nnet3_1_1NaturalGradientPerElementScaleComponent.html
[1165]: namespacekaldi_1_1nnet3.html
[1166]: structkaldi_1_1PlpOptions.html
[1167]: namespacekaldi.html
[1168]: structkaldi_1_1kws__internal_1_1ThrSweepStats.html
[1169]: namespacekaldi_1_1kws__internal.html
[1170]: classkaldi_1_1nnet3_1_1Component.html
[1171]: namespacekaldi_1_1nnet3.html
[1172]: classkaldi_1_1Fmpe.html
[1173]: namespacekaldi.html
[1174]: classkaldi_1_1nnet3_1_1NaturalGradientRepeatedAffineComponent.html
[1175]: namespacekaldi_1_1nnet3.html
[1176]: classkaldi_1_1nnet2_1_1PnormComponent.html
[1177]: namespacekaldi_1_1nnet2.html
[1178]: classkaldi_1_1TidToTstateMapper.html
[1179]: namespacekaldi.html
[1180]: structkaldi_1_1nnet1_1_1Component.html
[1181]: namespacekaldi_1_1nnet1.html
[1182]: structkaldi_1_1FmpeOptions.html
[1183]: namespacekaldi.html
[1184]: classfst_1_1NaturalLess_3_01CompactLatticeWeightTpl_3_01LatticeWeightTpl_3_01double_01_4_00_
01int32_01_4_01_4.html
[1185]: namespacefst.html
[1186]: classkaldi_1_1nnet3_1_1PnormComponent.html
[1187]: namespacekaldi_1_1nnet3.html
[1188]: classkaldi_1_1nnet3_1_1TimeHeightConvolutionComponent.html
[1189]: namespacekaldi_1_1nnet3.html
[1190]: classkaldi_1_1nnet3_1_1ComponentPrecomputedIndexes.html
[1191]: namespacekaldi_1_1nnet3.html
[1192]: structkaldi_1_1FmpeStats.html
[1193]: namespacekaldi.html
[1194]: classfst_1_1NaturalLess_3_01CompactLatticeWeightTpl_3_01LatticeWeightTpl_3_01float_01_4_00_0
1int32_01_4_01_4.html
[1195]: namespacefst.html
[1196]: structkaldi_1_1RefineClusterer_1_1point__info.html
[1197]: namespacekaldi.html
[1198]: classkaldi_1_1Timer.html
[1199]: namespacekaldi.html
[1200]: classfst_1_1ComposeDeterministicOnDemandFst.html
[1201]: namespacefst.html
[1202]: structkaldi_1_1FmpeUpdateOptions.html
[1203]: namespacekaldi.html
[1204]: classfst_1_1NaturalLess_3_01CompactLatticeWeightTpl_3_01LatticeWeightTpl_3_01FloatType_01_4_
00_01IntType_01_4_01_4.html
[1205]: namespacefst.html
[1206]: structkaldi_1_1nnet3_1_1ComputationRenumberer_1_1PointerCompare.html
[1207]: namespacekaldi_1_1nnet3.html
[1208]: classkaldi_1_1SimpleDecoder_1_1Token.html
[1209]: namespacekaldi.html
[1210]: structkaldi_1_1PrunedCompactLatticeComposer_1_1ComposedStateInfo.html
[1211]: namespacekaldi.html
[1212]: classkaldi_1_1nnet3_1_1ForwardingDescriptor.html
[1213]: namespacekaldi_1_1nnet3.html
[1214]: classfst_1_1NaturalLess_3_01LatticeWeightTpl_3_01double_01_4_01_4.html
[1215]: namespacefst.html
[1216]: classkaldi_1_1PosteriorHolder.html
[1217]: namespacekaldi.html
[1218]: classkaldi_1_1FasterDecoder_1_1Token.html
[1219]: namespacekaldi.html
[1220]: structkaldi_1_1ComposeLatticePrunedOptions.html
[1221]: namespacekaldi.html
[1222]: structkaldi_1_1LatticeBiglmFasterDecoder_1_1ForwardLink.html
[1223]: namespacekaldi.html
[1224]: classfst_1_1NaturalLess_3_01LatticeWeightTpl_3_01float_01_4_01_4.html
[1225]: namespacefst.html
[1226]: classkaldi_1_1nnet2_1_1PowerComponent.html
[1227]: namespacekaldi_1_1nnet2.html
[1228]: structkaldi_1_1LatticeBiglmFasterDecoder_1_1Token.html
[1229]: namespacekaldi.html
[1230]: classkaldi_1_1nnet3_1_1CompositeComponent.html
[1231]: namespacekaldi_1_1nnet3.html
[1232]: structkaldi_1_1decoder_1_1ForwardLink.html
[1233]: namespacekaldi_1_1decoder.html
[1234]: classfst_1_1NaturalLess_3_01LatticeWeightTpl_3_01FloatType_01_4_01_4.html
[1235]: namespacefst.html
[1236]: classkaldi_1_1nnet3_1_1RestrictedAttentionComponent_1_1PrecomputedIndexes.html
[1237]: namespacekaldi_1_1nnet3.html
[1238]: structkaldi_1_1LatticeSimpleDecoder_1_1Token.html
[1239]: namespacekaldi.html
[1240]: classkaldi_1_1CompressedAffineXformStats.html
[1241]: namespacekaldi.html
[1242]: structkaldi_1_1LatticeSimpleDecoder_1_1ForwardLink.html
[1243]: namespacekaldi.html
[1244]: structkaldi_1_1NccfInfo.html
[1245]: namespacekaldi.html
[1246]: classkaldi_1_1nnet3_1_1TimeHeightConvolutionComponent_1_1PrecomputedIndexes.html
[1247]: namespacekaldi_1_1nnet3.html
[1248]: classkaldi_1_1BiglmFasterDecoder_1_1Token.html
[1249]: namespacekaldi.html
[1250]: classkaldi_1_1CompressedMatrix.html
[1251]: namespacekaldi.html
[1252]: structkaldi_1_1FrameExtractionOptions.html
[1253]: namespacekaldi.html
[1254]: structkaldi_1_1nnet3_1_1NetworkNode.html
[1255]: namespacekaldi_1_1nnet3.html
[1256]: classkaldi_1_1nnet3_1_1TdnnComponent_1_1PrecomputedIndexes.html
[1257]: namespacekaldi_1_1nnet3.html
[1258]: classkaldi_1_1TokenHolder.html
[1259]: namespacekaldi.html
[1260]: classkaldi_1_1nnet3_1_1ComputationAnalysis.html
[1261]: namespacekaldi_1_1nnet3.html
[1262]: structkaldi_1_1nnet2_1_1DiscriminativeExampleSplitter_1_1FrameInfo.html
[1263]: namespacekaldi_1_1nnet2.html
[1264]: structrnnlm_1_1neuron.html
[1265]: namespacernnlm.html
[1266]: structkaldi_1_1nnet3_1_1NnetComputation_1_1PrecomputedIndexesInfo.html
[1267]: namespacekaldi_1_1nnet3.html
[1268]: structkaldi_1_1LatticeIncrementalDecoderTpl_1_1TokenList.html
[1269]: namespacekaldi.html
[1270]: classkaldi_1_1nnet3_1_1ComputationCache.html
[1271]: namespacekaldi_1_1nnet3.html
[1272]: structkaldi_1_1OnlineSilenceWeighting_1_1FrameInfo.html
[1273]: namespacekaldi.html
[1274]: structkaldi_1_1NGram.html
[1275]: namespacekaldi.html
[1276]: structkaldi_1_1ProcessPitchOptions.html
[1277]: namespacekaldi.html
[1278]: structkaldi_1_1LatticeSimpleDecoder_1_1TokenList.html
[1279]: namespacekaldi.html
[1280]: classkaldi_1_1nnet3_1_1ComputationChecker.html
[1281]: namespacekaldi_1_1nnet3.html
[1282]: classkaldi_1_1nnet1_1_1FramePoolingComponent.html
[1283]: namespacekaldi_1_1nnet1.html
[1284]: classkaldi_1_1nnet2_1_1Nnet.html
[1285]: namespacekaldi_1_1nnet2.html
[1286]: classkaldi_1_1Profiler.html
[1287]: namespacekaldi.html
[1288]: structkaldi_1_1LatticeBiglmFasterDecoder_1_1TokenList.html
[1289]: namespacekaldi.html
[1290]: classkaldi_1_1nnet3_1_1ComputationExpander.html
[1291]: namespacekaldi_1_1nnet3.html
[1292]: structfst_1_1GrammarFst_1_1FstInstance.html
[1293]: namespacefst.html
[1294]: classkaldi_1_1nnet3_1_1Nnet.html
[1295]: namespacekaldi_1_1nnet3.html
[1296]: classkaldi_1_1ProfileStats.html
[1297]: namespacekaldi.html
[1298]: structkaldi_1_1LatticeFasterDecoderTpl_1_1TokenList.html
[1299]: namespacekaldi.html
[1300]: structkaldi_1_1nnet3_1_1ComputationGraph.html
[1301]: namespacekaldi_1_1nnet3.html
[1302]: classkaldi_1_1FullGmm.html
[1303]: namespacekaldi.html
[1304]: classkaldi_1_1nnet1_1_1Nnet.html
[1305]: namespacekaldi_1_1nnet1.html
[1306]: structkaldi_1_1ProfileStats_1_1ProfileStatsEntry.html
[1307]: namespacekaldi.html
[1308]: classkaldi_1_1TokenVectorHolder.html
[1309]: namespacekaldi.html
[1310]: classkaldi_1_1nnet3_1_1ComputationGraphBuilder.html
[1311]: namespacekaldi_1_1nnet3.html
[1312]: classkaldi_1_1FullGmmNormal.html
[1313]: namespacekaldi.html
[1314]: classkaldi_1_1nnet3_1_1NnetBatchComputer.html
[1315]: namespacekaldi_1_1nnet3.html
[1316]: classkaldi_1_1PrunedCompactLatticeComposer.html
[1317]: namespacekaldi.html
[1318]: classkaldi_1_1TpMatrix.html
[1319]: namespacekaldi.html
[1320]: structkaldi_1_1nnet3_1_1NnetBatchComputer_1_1ComputationGroupInfo.html
[1321]: namespacekaldi_1_1nnet3.html
[1322]: structkaldi_1_1nnet3_1_1NnetBatchComputerOptions.html
[1323]: namespacekaldi_1_1nnet3.html
[1324]: classfst_1_1PruneSpecialClass.html
[1325]: namespacefst.html
[1326]: classkaldi_1_1TrainingGraphCompiler.html
[1327]: namespacekaldi.html
[1328]: structkaldi_1_1nnet3_1_1NnetBatchComputer_1_1ComputationGroupKey.html
[1329]: namespacekaldi_1_1nnet3.html
[1330]: classkaldi_1_1nnet3_1_1NnetBatchDecoder.html
[1331]: namespacekaldi_1_1nnet3.html
[1332]: classfst_1_1PushSpecialClass.html
[1333]: namespacefst.html
[1334]: structkaldi_1_1TrainingGraphCompilerOptions.html
[1335]: namespacekaldi.html
[1336]: structkaldi_1_1nnet3_1_1NnetBatchComputer_1_1ComputationGroupKeyHasher.html
[1337]: namespacekaldi_1_1nnet3.html
[1338]: structkaldi_1_1MinimumBayesRisk_1_1GammaCompare.html
[1339]: namespacekaldi.html
[1340]: classkaldi_1_1nnet3_1_1NnetBatchInference.html
[1341]: namespacekaldi_1_1nnet3.html
[1342]: classkaldi_1_1TransitionModel.html
[1343]: namespacekaldi.html
[1344]: classkaldi_1_1nnet3_1_1ComputationLoopedOptimizer.html
[1345]: namespacekaldi_1_1nnet3.html
[1346]: classkaldi_1_1GaussClusterable.html
[1347]: namespacekaldi.html
[1348]: classkaldi_1_1nnet3_1_1NnetChainComputeProb.html
[1349]: namespacekaldi_1_1nnet3.html
[1350]: classkaldi_1_1TreeClusterer.html
[1351]: namespacekaldi.html
[1352]: classkaldi_1_1nnet3_1_1ComputationRenumberer.html
[1353]: namespacekaldi_1_1nnet3.html
[1354]: structkaldi_1_1GaussInfo.html
[1355]: namespacekaldi.html
[1356]: structkaldi_1_1nnet3_1_1NnetChainExample.html
[1357]: namespacekaldi_1_1nnet3.html
[1358]: classkaldi_1_1Questions.html
[1359]: namespacekaldi.html
[1360]: structkaldi_1_1TreeClusterOptions.html
[1361]: namespacekaldi.html
[1362]: structkaldi_1_1nnet3_1_1ComputationRequest.html
[1363]: namespacekaldi_1_1nnet3.html
[1364]: classkaldi_1_1GaussPostHolder.html
[1365]: namespacekaldi.html
[1366]: structkaldi_1_1nnet3_1_1NnetChainExampleStructureCompare.html
[1367]: namespacekaldi_1_1nnet3.html
[1368]: structkaldi_1_1QuestionsForKey.html
[1369]: namespacekaldi.html
[1370]: classkaldi_1_1TreeRenderer.html
[1371]: namespacekaldi.html
[1372]: structkaldi_1_1nnet3_1_1ComputationRequestHasher.html
[1373]: namespacekaldi_1_1nnet3.html
[1374]: structkaldi_1_1nnet3_1_1GeneralDescriptor.html
[1375]: namespacekaldi_1_1nnet3.html
[1376]: structkaldi_1_1nnet3_1_1NnetChainExampleStructureHasher.html
[1377]: namespacekaldi_1_1nnet3.html
[1378]: classfst_1_1TrivialFactorWeightFst.html
[1379]: namespacefst.html
[1380]: structkaldi_1_1nnet3_1_1ComputationRequestPtrEqual.html
[1381]: namespacekaldi_1_1nnet3.html
[1382]: classkaldi_1_1nnet3_1_1GeneralDropoutComponent.html
[1383]: namespacekaldi_1_1nnet3.html
[1384]: structkaldi_1_1nnet3_1_1NnetChainSupervision.html
[1385]: namespacekaldi_1_1nnet3.html
[1386]: classfst_1_1internal_1_1TrivialFactorWeightFstImpl.html
[1387]: namespacefst_1_1internal.html
[1388]: classkaldi_1_1LatticePhoneAligner_1_1ComputationState.html
[1389]: namespacekaldi.html
[1390]: classkaldi_1_1nnet3_1_1GeneralDropoutComponentPrecomputedIndexes.html
[1391]: namespacekaldi_1_1nnet3.html
[1392]: classkaldi_1_1nnet3_1_1NnetChainTrainer.html
[1393]: namespacekaldi_1_1nnet3.html
[1394]: structfst_1_1RandFstOptions.html
[1395]: namespacefst.html
[1396]: structfst_1_1TrivialFactorWeightOptions.html
[1397]: namespacefst.html
[1398]: classkaldi_1_1LatticeLexiconWordAligner_1_1ComputationState.html
[1399]: namespacekaldi.html
[1400]: classkaldi_1_1GeneralMatrix.html
[1401]: namespacekaldi.html
[1402]: structkaldi_1_1nnet3_1_1NnetChainTrainingOptions.html
[1403]: namespacekaldi_1_1nnet3.html
[1404]: classkaldi_1_1RandomAccessTableReader.html
[1405]: namespacekaldi.html
[1406]: structkaldi_1_1LatticeLexiconWordAligner_1_1Tuple.html
[1407]: namespacekaldi.html
[1408]: classkaldi_1_1LatticeWordAligner_1_1ComputationState.html
[1409]: namespacekaldi.html
[1410]: classkaldi_1_1GenericHolder.html
[1411]: namespacekaldi.html
[1412]: structkaldi_1_1nnet2_1_1NnetCombineAconfig.html
[1413]: namespacekaldi_1_1nnet2.html
[1414]: classkaldi_1_1RandomAccessTableReaderArchiveImplBase.html
[1415]: namespacekaldi.html
[1416]: structkaldi_1_1LatticePhoneAligner_1_1Tuple.html
[1417]: namespacekaldi.html
[1418]: classkaldi_1_1nnet3_1_1ComputationStepsComputer.html
[1419]: namespacekaldi_1_1nnet3.html
[1420]: structkaldi_1_1CompressedMatrix_1_1GlobalHeader.html
[1421]: namespacekaldi.html
[1422]: structkaldi_1_1nnet2_1_1NnetCombineConfig.html
[1423]: namespacekaldi_1_1nnet2.html
[1424]: classkaldi_1_1RandomAccessTableReaderDSortedArchiveImpl.html
[1425]: namespacekaldi.html
[1426]: structkaldi_1_1LatticeWordAligner_1_1Tuple.html
[1427]: namespacekaldi.html
[1428]: classkaldi_1_1nnet3_1_1ComputationVariables.html
[1429]: namespacekaldi_1_1nnet3.html
[1430]: classfst_1_1GrammarFst.html
[1431]: namespacefst.html
[1432]: structkaldi_1_1nnet2_1_1NnetCombineFastConfig.html
[1433]: namespacekaldi_1_1nnet2.html
[1434]: classkaldi_1_1RandomAccessTableReaderImplBase.html
[1435]: namespacekaldi.html
[1436]: structkaldi_1_1TransitionModel_1_1Tuple.html
[1437]: namespacekaldi.html
[1438]: classkaldi_1_1ComputeNormalizersClass.html
[1439]: namespacekaldi.html
[1440]: structfst_1_1GrammarFstArc.html
[1441]: namespacefst.html
[1442]: structkaldi_1_1nnet3_1_1NnetComputation.html
[1443]: namespacekaldi_1_1nnet3.html
[1444]: classkaldi_1_1RandomAccessTableReaderMapped.html
[1445]: namespacekaldi.html
[1446]: structkaldi_1_1LatticeWordAligner_1_1TupleEqual.html
[1447]: namespacekaldi.html
[1448]: classkaldi_1_1ConfigLine.html
[1449]: namespacekaldi.html
[1450]: classfst_1_1GrammarFstPreparer.html
[1451]: namespacefst.html
[1452]: structkaldi_1_1nnet3_1_1NnetComputationPrintInserter.html
[1453]: namespacekaldi_1_1nnet3.html
[1454]: classkaldi_1_1RandomAccessTableReaderScriptImpl.html
[1455]: namespacekaldi.html
[1456]: structkaldi_1_1LatticeLexiconWordAligner_1_1TupleEqual.html
[1457]: namespacekaldi.html
[1458]: classkaldi_1_1nnet3_1_1ConstantComponent.html
[1459]: namespacekaldi_1_1nnet3.html
[1460]: structkaldi_1_1nnet3_1_1NnetComputeOptions.html
[1461]: namespacekaldi_1_1nnet3.html
[1462]: classkaldi_1_1RandomAccessTableReaderSortedArchiveImpl.html
[1463]: namespacekaldi.html
[1464]: structkaldi_1_1LatticePhoneAligner_1_1TupleEqual.html
[1465]: namespacekaldi.html
[1466]: classkaldi_1_1ConstantEventMap.html
[1467]: namespacekaldi.html
[1468]: classkaldi_1_1nnet3_1_1NnetComputeProb.html
[1469]: namespacekaldi_1_1nnet3.html
[1470]: classkaldi_1_1RandomAccessTableReaderUnsortedArchiveImpl.html
[1471]: namespacekaldi.html
[1472]: structkaldi_1_1LatticeLexiconWordAligner_1_1TupleHash.html
[1473]: namespacekaldi.html
[1474]: classkaldi_1_1nnet3_1_1ConstantFunctionComponent.html
[1475]: namespacekaldi_1_1nnet3.html
[1476]: structkaldi_1_1HashList_1_1HashBucket.html
[1477]: namespacekaldi.html
[1478]: structkaldi_1_1nnet3_1_1NnetComputeProbOptions.html
[1479]: namespacekaldi_1_1nnet3.html
[1480]: classkaldi_1_1nnet2_1_1RandomComponent.html
[1481]: namespacekaldi_1_1nnet2.html
[1482]: structkaldi_1_1LatticePhoneAligner_1_1TupleHash.html
[1483]: namespacekaldi.html
[1484]: classkaldi_1_1nnet3_1_1ConstantSumDescriptor.html
[1485]: namespacekaldi_1_1nnet3.html
[1486]: classkaldi_1_1HashList.html
[1487]: namespacekaldi.html
[1488]: classkaldi_1_1nnet2_1_1NnetComputer.html
[1489]: namespacekaldi_1_1nnet2.html
[1490]: classkaldi_1_1nnet3_1_1RandomComponent.html
[1491]: namespacekaldi_1_1nnet3.html
[1492]: structkaldi_1_1LatticeWordAligner_1_1TupleHash.html
[1493]: namespacekaldi.html
[1494]: classkaldi_1_1ConstArpaLm.html
[1495]: namespacekaldi.html
[1496]: classkaldi_1_1nnet1_1_1HiddenSoftmax.html
[1497]: namespacekaldi_1_1nnet1.html
[1498]: classkaldi_1_1nnet3_1_1NnetComputer.html
[1499]: namespacekaldi_1_1nnet3.html
[1500]: classkaldi_1_1nnet1_1_1RandomizerMask.html
[1501]: namespacekaldi_1_1nnet1.html
[1502]: classkaldi_1_1TwvMetrics.html
[1503]: namespacekaldi.html
[1504]: classkaldi_1_1ConstArpaLmBuilder.html
[1505]: namespacekaldi.html
[1506]: structkaldi_1_1HmmCacheHash.html
[1507]: namespacekaldi.html
[1508]: classkaldi_1_1nnet3_1_1NnetComputerFromEg.html
[1509]: namespacekaldi_1_1nnet3.html
[1510]: structkaldi_1_1RandomState.html
[1511]: namespacekaldi.html
[1512]: structkaldi_1_1TwvMetricsOptions.html
[1513]: namespacekaldi.html
[1514]: classkaldi_1_1ConstArpaLmDeterministicFst.html
[1515]: namespacekaldi.html
[1516]: structkaldi_1_1HmmTopology_1_1HmmState.html
[1517]: namespacekaldi.html
[1518]: structkaldi_1_1nnet1_1_1NnetDataRandomizerOptions.html
[1519]: namespacekaldi_1_1nnet1.html
[1520]: classkaldi_1_1nnet1_1_1Rbm.html
[1521]: namespacekaldi_1_1nnet1.html
[1522]: classkaldi_1_1TwvMetricsStats.html
[1523]: namespacekaldi.html
[1524]: classkaldi_1_1ConstIntegerSet.html
[1525]: namespacekaldi.html
[1526]: classkaldi_1_1HmmTopology.html
[1527]: namespacekaldi.html
[1528]: classkaldi_1_1nnet3_1_1NnetDiscriminativeComputeObjf.html
[1529]: namespacekaldi_1_1nnet3.html
[1530]: classkaldi_1_1nnet1_1_1RbmBase.html
[1531]: namespacekaldi_1_1nnet1.html
[1532]: classkaldi_1_1ContextDependency.html
[1533]: namespacekaldi.html
[1534]: structkaldi_1_1HtkHeader.html
[1535]: namespacekaldi.html
[1536]: structkaldi_1_1nnet3_1_1NnetDiscriminativeExample.html
[1537]: namespacekaldi_1_1nnet3.html
[1538]: structkaldi_1_1nnet1_1_1RbmTrainOptions.html
[1539]: namespacekaldi_1_1nnet1.html
[1540]: classkaldi_1_1ContextDependencyInterface.html
[1541]: namespacekaldi.html
[1542]: classkaldi_1_1HtkMatrixHolder.html
[1543]: namespacekaldi.html
[1544]: structkaldi_1_1nnet3_1_1NnetDiscriminativeExampleStructureCompare.html
[1545]: namespacekaldi_1_1nnet3.html
[1546]: structkaldi_1_1RecognizedWord.html
[1547]: namespacekaldi.html
[1548]: structkaldi_1_1UbmClusteringOptions.html
[1549]: namespacekaldi.html
[1550]: classkaldi_1_1nnet2_1_1Convolutional1dComponent.html
[1551]: namespacekaldi_1_1nnet2.html
[1552]: structkaldi_1_1HTransducerConfig.html
[1553]: namespacekaldi.html
[1554]: structkaldi_1_1nnet3_1_1NnetDiscriminativeExampleStructureHasher.html
[1555]: namespacekaldi_1_1nnet3.html
[1556]: classkaldi_1_1nnet2_1_1RectifiedLinearComponent.html
[1557]: namespacekaldi_1_1nnet2.html
[1558]: classfst_1_1UnweightedNgramFst.html
[1559]: namespacefst.html
[1560]: classkaldi_1_1nnet1_1_1ConvolutionalComponent.html
[1561]: namespacekaldi_1_1nnet1.html
[1562]: structkaldi_1_1nnet3_1_1NnetDiscriminativeOptions.html
[1563]: namespacekaldi_1_1nnet3.html
[1564]: classkaldi_1_1nnet3_1_1RectifiedLinearComponent.html
[1565]: namespacekaldi_1_1nnet3.html
[1566]: classkaldi_1_1nnet3_1_1UpdatableComponent.html
[1567]: namespacekaldi_1_1nnet3.html
[1568]: classkaldi_1_1nnet3_1_1ConvolutionComponent.html
[1569]: namespacekaldi_1_1nnet3.html
[1570]: structkaldi_1_1nnet2_1_1NnetDiscriminativeStats.html
[1571]: namespacekaldi_1_1nnet2.html
[1572]: classkaldi_1_1nnet1_1_1RecurrentComponent.html
[1573]: namespacekaldi_1_1nnet1.html
[1574]: classkaldi_1_1nnet1_1_1UpdatableComponent.html
[1575]: namespacekaldi_1_1nnet1.html
[1576]: structkaldi_1_1nnet3_1_1time__height__convolution_1_1ConvolutionComputation.html
[1577]: namespacekaldi_1_1nnet3_1_1time__height__convolution.html
[1578]: structfst_1_1IdentityFunction.html
[1579]: namespacefst.html
[1580]: structkaldi_1_1nnet3_1_1NnetDiscriminativeSupervision.html
[1581]: namespacekaldi_1_1nnet3.html
[1582]: classkaldi_1_1RecyclingVector.html
[1583]: namespacekaldi.html
[1584]: classkaldi_1_1nnet2_1_1UpdatableComponent.html
[1585]: namespacekaldi_1_1nnet2.html
[1586]: structkaldi_1_1nnet3_1_1time__height__convolution_1_1ConvolutionComputationIo.html
[1587]: namespacekaldi_1_1nnet3_1_1time__height__convolution.html
[1588]: structkaldi_1_1nnet3_1_1ImageAugmentationConfig.html
[1589]: namespacekaldi_1_1nnet3.html
[1590]: classkaldi_1_1nnet3_1_1NnetDiscriminativeTrainer.html
[1591]: namespacekaldi_1_1nnet3.html
[1592]: classkaldi_1_1RefineClusterer.html
[1593]: namespacekaldi.html
[1594]: classkaldi_1_1UpdatePhoneVectorsClass.html
[1595]: namespacekaldi.html
[1596]: structkaldi_1_1nnet3_1_1time__height__convolution_1_1ConvolutionComputationOptions.html
[1597]: namespacekaldi_1_1nnet3_1_1time__height__convolution.html
[1598]: classImplToFst.html
[1599]: structkaldi_1_1nnet2_1_1NnetDiscriminativeUpdateOptions.html
[1600]: namespacekaldi_1_1nnet2.html
[1601]: structkaldi_1_1RefineClustersOptions.html
[1602]: namespacekaldi.html
[1603]: classkaldi_1_1UpdateWClass.html
[1604]: namespacekaldi.html
[1605]: structkaldi_1_1nnet3_1_1time__height__convolution_1_1ConvolutionModel.html
[1606]: namespacekaldi_1_1nnet3_1_1time__height__convolution.html
[1607]: structkaldi_1_1nnet3_1_1Index.html
[1608]: namespacekaldi_1_1nnet3.html
[1609]: classkaldi_1_1nnet2_1_1NnetDiscriminativeUpdater.html
[1610]: namespacekaldi_1_1nnet2.html
[1611]: classkaldi_1_1RegressionTree.html
[1612]: namespacekaldi.html
[1613]: structkaldi_1_1nnet3_1_1NnetBatchInference_1_1UtteranceInfo.html
[1614]: namespacekaldi_1_1nnet3.html
[1615]: structkaldi_1_1nnet3_1_1time__height__convolution_1_1ConvolutionComputation_1_1ConvolutionSt
ep.html
[1616]: namespacekaldi_1_1nnet3_1_1time__height__convolution.html
[1617]: structkaldi_1_1nnet3_1_1IndexHasher.html
[1618]: namespacekaldi_1_1nnet3.html
[1619]: classkaldi_1_1nnet2_1_1NnetEnsembleTrainer.html
[1620]: namespacekaldi_1_1nnet2.html
[1621]: classkaldi_1_1RegtreeFmllrDiagGmm.html
[1622]: namespacekaldi.html
[1623]: structkaldi_1_1nnet3_1_1NnetBatchDecoder_1_1UtteranceInput.html
[1624]: namespacekaldi_1_1nnet3.html
[1625]: classkaldi_1_1nnet1_1_1CopyComponent.html
[1626]: namespacekaldi_1_1nnet1.html
[1627]: structkaldi_1_1nnet3_1_1IndexLessNxt.html
[1628]: namespacekaldi_1_1nnet3.html
[1629]: structkaldi_1_1nnet2_1_1NnetEnsembleTrainerConfig.html
[1630]: namespacekaldi_1_1nnet2.html
[1631]: classkaldi_1_1RegtreeFmllrDiagGmmAccs.html
[1632]: namespacekaldi.html
[1633]: structkaldi_1_1nnet3_1_1NnetBatchDecoder_1_1UtteranceOutput.html
[1634]: namespacekaldi_1_1nnet3.html
[1635]: structkaldi_1_1CountStats.html
[1636]: namespacekaldi.html
[1637]: classkaldi_1_1nnet3_1_1IndexSet.html
[1638]: namespacekaldi_1_1nnet3.html
[1639]: structkaldi_1_1nnet2_1_1NnetExample.html
[1640]: namespacekaldi_1_1nnet2.html
[1641]: structkaldi_1_1RegtreeFmllrOptions.html
[1642]: namespacekaldi.html
[1643]: classkaldi_1_1nnet3_1_1UtteranceSplitter.html
[1644]: namespacekaldi_1_1nnet3.html
[1645]: classkaldi_1_1CovarianceStats.html
[1646]: namespacekaldi.html
[1647]: structkaldi_1_1nnet3_1_1IndexVectorHasher.html
[1648]: namespacekaldi_1_1nnet3.html
[1649]: structkaldi_1_1nnet3_1_1NnetExample.html
[1650]: namespacekaldi_1_1nnet3.html
[1651]: classkaldi_1_1RegtreeMllrDiagGmm.html
[1652]: namespacekaldi.html
[1653]: classrnnlm_1_1CRnnLM.html
[1654]: namespacernnlm.html
[1655]: classkaldi_1_1Input.html
[1656]: namespacekaldi.html
[1657]: classkaldi_1_1nnet2_1_1NnetExampleBackgroundReader.html
[1658]: namespacekaldi_1_1nnet2.html
[1659]: classkaldi_1_1RegtreeMllrDiagGmmAccs.html
[1660]: namespacekaldi.html
[1661]: structkaldi_1_1CuAllocatorOptions.html
[1662]: namespacekaldi.html
[1663]: classkaldi_1_1InputImplBase.html
[1664]: namespacekaldi.html
[1665]: structkaldi_1_1nnet3_1_1NnetExampleStructureCompare.html
[1666]: namespacekaldi_1_1nnet3.html
[1667]: structkaldi_1_1RegtreeMllrOptions.html
[1668]: namespacekaldi.html
[1669]: structkaldi_1_1VadEnergyOptions.html
[1670]: namespacekaldi.html
[1671]: classkaldi_1_1CuArray.html
[1672]: namespacekaldi.html
[1673]: unionkaldi_1_1Int32AndFloat.html
[1674]: namespacekaldi.html
[1675]: structkaldi_1_1nnet3_1_1NnetExampleStructureHasher.html
[1676]: namespacekaldi_1_1nnet3.html
[1677]: classfst_1_1RemoveEpsLocalClass.html
[1678]: namespacefst.html
[1679]: classkaldi_1_1nnet3_1_1VariableMergingOptimizer.html
[1680]: namespacekaldi_1_1nnet3.html
[1681]: classkaldi_1_1CuArrayBase.html
[1682]: namespacekaldi.html
[1683]: structkaldi_1_1Int32IsZero.html
[1684]: namespacekaldi.html
[1685]: structkaldi_1_1nnet2_1_1NnetFixConfig.html
[1686]: namespacekaldi_1_1nnet2.html
[1687]: classfst_1_1RemoveSomeInputSymbolsMapper.html
[1688]: namespacefst.html
[1689]: classkaldi_1_1Vector.html
[1690]: namespacekaldi.html
[1691]: classkaldi_1_1CuBlockMatrix.html
[1692]: namespacekaldi.html
[1693]: structInt32Pair.html
[1694]: structkaldi_1_1nnet3_1_1NnetGenerationOptions.html
[1695]: namespacekaldi_1_1nnet3.html
[1696]: classkaldi_1_1nnet3_1_1RepeatedAffineComponent.html
[1697]: namespacekaldi_1_1nnet3.html
[1698]: classkaldi_1_1VectorBase.html
[1699]: namespacekaldi.html
[1700]: structCuBlockMatrixData__.html
[1701]: classkaldi_1_1Interval.html
[1702]: namespacekaldi.html
[1703]: structkaldi_1_1nnet3_1_1NnetInferenceTask.html
[1704]: namespacekaldi_1_1nnet3.html
[1705]: classkaldi_1_1nnet3_1_1ReplaceIndexForwardingDescriptor.html
[1706]: namespacekaldi_1_1nnet3.html
[1707]: classkaldi_1_1VectorClusterable.html
[1708]: namespacekaldi.html
[1709]: classkaldi_1_1CuCompressedMatrix.html
[1710]: namespacekaldi.html
[1711]: structkaldi_1_1nnet3_1_1ExampleMergingConfig_1_1IntSet.html
[1712]: namespacekaldi_1_1nnet3.html
[1713]: structkaldi_1_1nnet3_1_1NnetIo.html
[1714]: namespacekaldi_1_1nnet3.html
[1715]: classkaldi_1_1nnet1_1_1Rescale.html
[1716]: namespacekaldi_1_1nnet1.html
[1717]: classfst_1_1StringRepository_1_1VectorEqual.html
[1718]: namespacefst.html
[1719]: classkaldi_1_1CuCompressedMatrixBase.html
[1720]: namespacekaldi.html
[1721]: classfst_1_1InverseContextFst.html
[1722]: namespacefst.html
[1723]: structkaldi_1_1nnet3_1_1NnetIoStructureCompare.html
[1724]: namespacekaldi_1_1nnet3.html
[1725]: classkaldi_1_1nnet3_1_1RestrictedAttentionComponent.html
[1726]: namespacekaldi_1_1nnet3.html
[1727]: classkaldi_1_1VectorFstToKwsLexicographicFstMapper.html
[1728]: namespacekaldi.html
[1729]: classkaldi_1_1CuMatrix.html
[1730]: namespacekaldi.html
[1731]: classfst_1_1InverseLeftBiphoneContextFst.html
[1732]: namespacefst.html
[1733]: structkaldi_1_1nnet3_1_1NnetIoStructureHasher.html
[1734]: namespacekaldi_1_1nnet3.html
[1735]: structkaldi_1_1ProfileStats_1_1ReverseSecondComparator.html
[1736]: namespacekaldi.html
[1737]: classfst_1_1VectorFstTplHolder.html
[1738]: namespacefst.html
[1739]: classkaldi_1_1CuMatrixBase.html
[1740]: namespacekaldi.html
[1741]: structkaldi_1_1nnet3_1_1IoSpecification.html
[1742]: namespacekaldi_1_1nnet3.html
[1743]: classkaldi_1_1nnet3_1_1NnetLdaStatsAccumulator.html
[1744]: namespacekaldi_1_1nnet3.html
[1745]: structfst_1_1ReweightPlusDefault.html
[1746]: namespacefst.html
[1747]: structkaldi_1_1VectorHasher.html
[1748]: namespacekaldi.html
[1749]: classkaldi_1_1CuPackedMatrix.html
[1750]: namespacekaldi.html
[1751]: structkaldi_1_1nnet3_1_1IoSpecificationHasher.html
[1752]: namespacekaldi_1_1nnet3.html
[1753]: structkaldi_1_1nnet2_1_1NnetLimitRankOpts.html
[1754]: namespacekaldi_1_1nnet2.html
[1755]: structfst_1_1ReweightPlusLogArc.html
[1756]: namespacefst.html
[1757]: classfst_1_1StringRepository_1_1VectorKey.html
[1758]: namespacefst.html
[1759]: classkaldi_1_1CuRand.html
[1760]: namespacekaldi.html
[1761]: structkaldi_1_1IvectorEstimationOptions.html
[1762]: namespacekaldi.html
[1763]: structkaldi_1_1nnet2_1_1NnetMixupConfig.html
[1764]: namespacekaldi_1_1nnet2.html
[1765]: classkaldi_1_1RnnlmDeterministicFst.html
[1766]: namespacekaldi.html
[1767]: classkaldi_1_1nnet1_1_1VectorRandomizer.html
[1768]: namespacekaldi_1_1nnet1.html
[1769]: classkaldi_1_1CuSparseMatrix.html
[1770]: namespacekaldi.html
[1771]: classkaldi_1_1IvectorExtractor.html
[1772]: namespacekaldi.html
[1773]: classkaldi_1_1nnet2_1_1NnetOnlineComputer.html
[1774]: namespacekaldi_1_1nnet2.html
[1775]: classkaldi_1_1nnet3_1_1RoundingForwardingDescriptor.html
[1776]: namespacekaldi_1_1nnet3.html
[1777]: structrnnlm_1_1vocab__word.html
[1778]: namespacernnlm.html
[1779]: classkaldi_1_1CuSpMatrix.html
[1780]: namespacekaldi.html
[1781]: classkaldi_1_1IvectorExtractorComputeDerivedVarsClass.html
[1782]: namespacekaldi.html
[1783]: structkaldi_1_1nnet3_1_1NnetOptimizeOptions.html
[1784]: namespacekaldi_1_1nnet3.html
[1785]: classkaldi_1_1nnet3_1_1RowOpsSplitter.html
[1786]: namespacekaldi_1_1nnet3.html
[1787]: classkaldi_1_1CuSubArray.html
[1788]: namespacekaldi.html
[1789]: structkaldi_1_1IvectorExtractorEstimationOptions.html
[1790]: namespacekaldi.html
[1791]: structkaldi_1_1nnet2_1_1NnetRescaleConfig.html
[1792]: namespacekaldi_1_1nnet2.html
[1793]: structkaldi_1_1RspecifierOptions.html
[1794]: namespacekaldi.html
[1795]: classkaldi_1_1CuSubMatrix.html
[1796]: namespacekaldi.html
[1797]: structkaldi_1_1IvectorExtractorOptions.html
[1798]: namespacekaldi.html
[1799]: classkaldi_1_1nnet2_1_1NnetRescaler.html
[1800]: namespacekaldi_1_1nnet2.html
[1801]: structkaldi_1_1TaskSequencer_1_1RunTaskArgsList.html
[1802]: namespacekaldi.html
[1803]: classkaldi_1_1WaveData.html
[1804]: namespacekaldi.html
[1805]: classkaldi_1_1CuSubVector.html
[1806]: namespacekaldi.html
[1807]: classkaldi_1_1IvectorExtractorStats.html
[1808]: namespacekaldi.html
[1809]: structkaldi_1_1nnet2_1_1NnetShrinkConfig.html
[1810]: namespacekaldi_1_1nnet2.html
[1811]: structkaldi_1_1WaveHeaderReadGofer.html
[1812]: namespacekaldi.html
[1813]: classkaldi_1_1CuTpMatrix.html
[1814]: namespacekaldi.html
[1815]: structkaldi_1_1IvectorExtractorStatsOptions.html
[1816]: namespacekaldi.html
[1817]: structkaldi_1_1nnet3_1_1NnetSimpleComputationOptions.html
[1818]: namespacekaldi_1_1nnet3.html
[1819]: classkaldi_1_1WaveHolder.html
[1820]: namespacekaldi.html
[1821]: classkaldi_1_1CuValue.html
[1822]: namespacekaldi.html
[1823]: classkaldi_1_1IvectorExtractorUpdateProjectionClass.html
[1824]: namespacekaldi.html
[1825]: structkaldi_1_1nnet3_1_1NnetSimpleLoopedComputationOptions.html
[1826]: namespacekaldi_1_1nnet3.html
[1827]: classkaldi_1_1ScalarClusterable.html
[1828]: namespacekaldi.html
[1829]: classkaldi_1_1WaveInfo.html
[1830]: namespacekaldi.html
[1831]: classkaldi_1_1CuVector.html
[1832]: namespacekaldi.html
[1833]: classkaldi_1_1IvectorExtractorUpdateWeightClass.html
[1834]: namespacekaldi.html
[1835]: structkaldi_1_1nnet2_1_1NnetSimpleTrainerConfig.html
[1836]: namespacekaldi_1_1nnet2.html
[1837]: classkaldi_1_1nnet3_1_1ScaleAndOffsetComponent.html
[1838]: namespacekaldi_1_1nnet3.html
[1839]: classkaldi_1_1WaveInfoHolder.html
[1840]: namespacekaldi.html
[1841]: classkaldi_1_1CuVectorBase.html
[1842]: namespacekaldi.html
[1843]: classkaldi_1_1IvectorExtractorUtteranceStats.html
[1844]: namespacekaldi.html
[1845]: classkaldi_1_1nnet2_1_1NnetStats.html
[1846]: namespacekaldi_1_1nnet2.html
[1847]: classkaldi_1_1nnet2_1_1ScaleComponent.html
[1848]: namespacekaldi_1_1nnet2.html
[1849]: classkaldi_1_1WordAlignedLatticeTester.html
[1850]: namespacekaldi.html
[1851]: classkaldi_1_1IvectorExtractTask.html
[1852]: namespacekaldi.html
[1853]: structkaldi_1_1nnet2_1_1NnetStatsConfig.html
[1854]: namespacekaldi_1_1nnet2.html
[1855]: classfst_1_1ScaleDeterministicOnDemandFst.html
[1856]: namespacefst.html
[1857]: classkaldi_1_1WordAlignLatticeLexiconInfo.html
[1858]: namespacekaldi.html
[1859]: classkaldi_1_1IvectorTask.html
[1860]: namespacekaldi.html
[1861]: classkaldi_1_1nnet3_1_1NnetTrainer.html
[1862]: namespacekaldi_1_1nnet3.html
[1863]: classkaldi_1_1Semaphore.html
[1864]: namespacekaldi.html
[1865]: structkaldi_1_1WordAlignLatticeLexiconOpts.html
[1866]: namespacekaldi.html
[1867]: classkaldi_1_1nnet2_1_1DctComponent.html
[1868]: namespacekaldi_1_1nnet2.html
[1869]: structkaldi_1_1nnet3_1_1NnetTrainerOptions.html
[1870]: namespacekaldi_1_1nnet3.html
[1871]: classkaldi_1_1nnet1_1_1SentenceAveragingComponent.html
[1872]: namespacekaldi_1_1nnet1.html
[1873]: structkaldi_1_1WordBoundaryInfo.html
[1874]: namespacekaldi.html
[1875]: classkaldi_1_1DecisionTreeSplitter.html
[1876]: namespacekaldi.html
[1877]: structkaldi_1_1nnet1_1_1NnetTrainOptions.html
[1878]: namespacekaldi_1_1nnet1.html
[1879]: classkaldi_1_1differentiable__transform_1_1SequenceTransform.html
[1880]: namespacekaldi_1_1differentiable__transform.html
[1881]: structkaldi_1_1WordBoundaryInfoNewOpts.html
[1882]: namespacekaldi.html
[1883]: classkaldi_1_1DecodableAmDiagGmm.html
[1884]: namespacekaldi.html
[1885]: classKaldiCompileTimeAssert.html
[1886]: classkaldi_1_1nnet2_1_1NnetUpdater.html
[1887]: namespacekaldi_1_1nnet2.html
[1888]: classkaldi_1_1SequentialTableReader.html
[1889]: namespacekaldi.html
[1890]: structkaldi_1_1WordBoundaryInfoOpts.html
[1891]: namespacekaldi.html
[1892]: classkaldi_1_1DecodableAmDiagGmmRegtreeFmllr.html
[1893]: namespacekaldi.html
[1894]: classKaldiCompileTimeAssert_3_01true_01_4.html
[1895]: structkaldi_1_1nnet2_1_1NnetWidenConfig.html
[1896]: namespacekaldi_1_1nnet2.html
[1897]: classkaldi_1_1SequentialTableReaderArchiveImpl.html
[1898]: namespacekaldi.html
[1899]: structkaldi_1_1ConstArpaLmBuilder_1_1WordsAndLmStatePairLessThan.html
[1900]: namespacekaldi.html
[1901]: classkaldi_1_1DecodableAmDiagGmmRegtreeMllr.html
[1902]: namespacekaldi.html
[1903]: classkaldi_1_1KaldiFatalError.html
[1904]: namespacekaldi.html
[1905]: structkaldi_1_1TreeClusterer_1_1Node.html
[1906]: namespacekaldi.html
[1907]: classkaldi_1_1SequentialTableReaderBackgroundImpl.html
[1908]: namespacekaldi.html
[1909]: structkaldi_1_1WspecifierOptions.html
[1910]: namespacekaldi.html
[1911]: classkaldi_1_1DecodableAmDiagGmmScaled.html
[1912]: namespacekaldi.html
[1913]: classkaldi_1_1KaldiObjectHolder.html
[1914]: namespacekaldi.html
[1915]: classkaldi_1_1nnet2_1_1NonlinearComponent.html
[1916]: namespacekaldi_1_1nnet2.html
[1917]: classkaldi_1_1SequentialTableReaderImplBase.html
[1918]: namespacekaldi.html
[1919]: classkaldi_1_1DecodableAmDiagGmmUnmapped.html
[1920]: namespacekaldi.html
[1921]: classkaldi_1_1KaldiRnnlmWrapper.html
[1922]: namespacekaldi.html
[1923]: classkaldi_1_1nnet3_1_1NonlinearComponent.html
[1924]: namespacekaldi_1_1nnet3.html
[1925]: classkaldi_1_1SequentialTableReaderScriptImpl.html
[1926]: namespacekaldi.html
[1927]: classkaldi_1_1nnet2_1_1DecodableAmNnet.html
[1928]: namespacekaldi_1_1nnet2.html
[1929]: structkaldi_1_1KaldiRnnlmWrapperOpts.html
[1930]: namespacekaldi.html
[1931]: classkaldi_1_1nnet3_1_1NoOpComponent.html
[1932]: namespacekaldi_1_1nnet3.html
[1933]: structkaldi_1_1Sgmm2FmllrConfig.html
[1934]: namespacekaldi.html
[1935]: classkaldi_1_1nnet1_1_1Xent.html
[1936]: namespacekaldi_1_1nnet1.html
[1937]: classkaldi_1_1nnet3_1_1DecodableAmNnetLoopedOnline.html
[1938]: namespacekaldi_1_1nnet3.html
[1939]: structkaldi_1_1nnet1_1_1Component_1_1key__value.html
[1940]: namespacekaldi_1_1nnet1.html
[1941]: classkaldi_1_1differentiable__transform_1_1NoOpTransform.html
[1942]: namespacekaldi_1_1differentiable__transform.html
[1943]: classkaldi_1_1Sgmm2FmllrGlobalParams.html
[1944]: namespacekaldi.html
[1945]: structkaldi_1_1nnet3_1_1BatchedXvectorComputer_1_1XvectorTask.html
[1946]: namespacekaldi_1_1nnet3.html
[1947]: classkaldi_1_1nnet2_1_1DecodableAmNnetParallel.html
[1948]: namespacekaldi_1_1nnet2.html
[1949]: classkaldi_1_1nnet1_1_1KlHmm.html
[1950]: namespacekaldi_1_1nnet1.html
[1951]: structkaldi_1_1OnlineProcessPitch_1_1NormalizationStats.html
[1952]: namespacekaldi.html
[1953]: classkaldi_1_1Sgmm2GauPost.html
[1954]: namespacekaldi.html
[1955]: classkaldi_1_1nnet3_1_1DecodableAmNnetSimple.html
[1956]: namespacekaldi_1_1nnet3.html
[1957]: classkMarkerMap.html
[1958]: classkaldi_1_1nnet3_1_1NormalizeComponent.html
[1959]: namespacekaldi_1_1nnet3.html
[1960]: structkaldi_1_1Sgmm2GauPostElement.html
[1961]: namespacekaldi.html
[1962]: #letter_a
[1963]: #letter_b
[1964]: #letter_c
[1965]: #letter_d
[1966]: #letter_e
[1967]: #letter_f
[1968]: #letter_g
[1969]: #letter_h
[1970]: #letter_i
[1971]: #letter_k
[1972]: #letter_l
[1973]: #letter_m
[1974]: #letter_n
[1975]: #letter_o
[1976]: #letter_p
[1977]: #letter_q
[1978]: #letter_r
[1979]: #letter_s
[1980]: #letter_t
[1981]: #letter_u
[1982]: #letter_v
[1983]: #letter_w
[1984]: #letter_x
[1985]: http://www.doxygen.org/index.html
