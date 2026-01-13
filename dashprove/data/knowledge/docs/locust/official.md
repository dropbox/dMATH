* Locust Documentation

# Locust Documentation[][1]

## Getting started[][2]

* [What is Locust?][3]
  
  * [Features][4]
  * [Name & background][5]
  * [Authors][6]
  * [License][7]
* [Installation][8]
  
  * [Using uvx (alternative)][9]
  * [Done!][10]
  * [Pre-release builds][11]
  * [Install for development][12]
  * [Troubleshooting installation][13]
* [Your first test][14]
  
  * [Locust’s web interface][15]
  * [Direct command line usage / headless][16]
  * [More options][17]
  * [Next steps][18]

## Writing Locust tests[][19]

* [Writing a locustfile][20]
  
  * [Auto-generating a locustfile][21]
  * [User class][22]
  * [Tasks][23]
  * [Events][24]
  * [HttpUser class][25]
  * [TaskSets][26]
  * [Examples][27]
  * [How to structure your test code][28]

## Running your Locust tests[][29]

* [Configuration][30]
* [Increasing the request rate][31]
* [Distributed load generation][32]
* [Running tests in a debugger][33]
* [Running in Docker][34]
* [Running without the web UI][35]

## Running with Locust Cloud[][36]

* [Locust Cloud Documentation][37]
  
  * [Running with Locust Cloud][38]
  * [First run][39]
  * [Locustfile and mock server][40]
  * [Passing options to Locust][41]
  * [Automated runs (CI)][42]
  * [Extra Python packages and files][43]
  * [View dashboard / previous test runs][44]
  * [Source IP Addresses][45]
* [Kubernetes Operator][46]
  
  * [Installation][47]
    
    * [Helm Charts][48]
    * [Manifest Files][49]
  * [Quickstart][50]
  * [LocustTest CRD Configuration][51]
    
    * [General][52]
    * [Locustfile source][53]
    * [Metadata][54]
    * [Per-role overrides][55]
    * [Extended][56]
  * [Examples][57]
    
    * [Inline locustfile][58]
    * [External ConfigMap locustfile][59]
    * [Custom Master/Worker pod configuration][60]
    * [Headless run][61]
  * [Upgrade][62]
    
    * [Helm][63]
  * [Uninstall][64]
    
    * [Helm][65]
    * [Manifest Files][66]

If you have questions or get stuck, feel free to reach out to us at [support@locust.cloud][67].

## Other functionalities[][68]

* [Custom load shapes][69]
  
  * [Example][70]
  * [Combining Users with different load profiles][71]
  * [Restricting which user types to spawn in each tick][72]
  * [Reusing common options in custom shapes][73]
* [Retrieve test statistics in CSV format][74]
* [Testing other systems/protocols][75]
  
  * [XML-RPC][76]
  * [gRPC][77]
  * [requests-based libraries/SDKs][78]
  * [REST][79]
  * [SocketIO][80]
  * [pytest][81]
  * [OpenAI][82]
  * [MQTT][83]
  * [Other examples][84]
* [Increase performance with a faster HTTP client][85]
  
  * [How to use FastHttpUser][86]
  * [Concurrency][87]
  * [REST][88]
  * [Connection Handling][89]
  * [API][90]
* [Event hooks][91]
  
  * [Request context][92]
  * [Adding Web Routes][93]
  * [Extending Web UI][94]
  * [Adding Authentication to the Web UI][95]
  * [Run a background greenlet][96]
  * [Parametrizing locustfiles][97]
  * [Test data management][98]
  * [More examples][99]
* [Logging][100]
  
  * [Options][101]
  * [Locust loggers][102]
* [Using Locust as a library][103]
  
  * [Skipping monkey patching][104]
  * [Full example][105]
* [OpenTelemetry Integration][106]
  
  * [Setup][107]
  * [Exporters][108]
  * [Auto Instrumentation][109]
  * [Example][110]
* [Third party extensions][111]
  
  * [Support for load testing other protocols, reporting etc][112]
  * [Report OTEL traces for requests][113]
  * [Automatically translate a browser recording (HAR-file) to a locustfile][114]
  * [Workers written in other languages than Python][115]
* [VS Code Extension][116]

## Further reading / knowledgebase[][117]

* [Developing and Documenting Locust][118]
* [Further reading / knowledgebase][119]
* [FAQ][120]

## API[][121]

* [API][122]
  
  * [User class][123]
  * [HttpUser class][124]
  * [HttpSession class][125]
  * [FastHttpUser class][126]
  * [MqttUser class][127]
  * [SocketIOUser class][128]
  * [FastHttpSession class][129]
  * [PostgresUser class][130]
  * [MongoDBUser class][131]
  * [MilvusUser class][132]
  * [DNSUser class][133]
  * [TaskSet class][134]
  * [task decorator][135]
  * [tag decorator][136]
  * [SequentialTaskSet class][137]
  * [Built in wait_time functions][138]
  * [Response class][139]
  * [ResponseContextManager class][140]
  * [Exceptions][141]
  * [Environment class][142]
  * [Event hooks][143]
  * [Runner classes][144]
  * [Web UI class][145]
  * [Other][146]

## Changelog[][147]

* [Changelog Highlights][148]
  
  * [2.42.6][149]
  * [2.42.5][150]
  * [2.42.4][151]
  * [2.42.3][152]
  * [2.42.2][153]
  * [2.42.1][154]
  * [2.42.0][155]
  * [2.41.6][156]
  * [2.41.5][157]
  * [2.41.4][158]
  * [2.41.3][159]
  * [2.41.2][160]
  * [2.41.1][161]
  * [2.41.0][162]
  * [2.40.5][163]
  * [2.40.4][164]
  * [2.40.3][165]
  * [2.40.2][166]
  * [2.40.1][167]
  * [2.40.0][168]
  * [2.39.1][169]
  * [2.39.0][170]
  * [2.38.1][171]
  * [2.38.0][172]
  * [2.37.14][173]
  * [2.37.13][174]
  * [2.37.12][175]
  * [2.37.11][176]
  * [2.37.10][177]
  * [2.37.9][178]
  * [2.37.8][179]
  * [2.37.7][180]
  * [2.37.6][181]
  * [2.37.5][182]
  * [2.37.4][183]
  * [2.37.3][184]
  * [2.37.2][185]
  * [2.37.1][186]
  * [2.37.0][187]
  * [2.36.3][188]
  * [2.36.2][189]
  * [2.36.1][190]
  * [2.36.0][191]
  * [2.35.0][192]
  * [2.34.1][193]
  * [2.34.0][194]
  * [2.33.2][195]
  * [2.33.1][196]
  * [2.33.0][197]
  * [2.32.10][198]
  * [2.32.9][199]
  * [2.32.8][200]
  * [2.32.7][201]
  * [2.32.6][202]
  * [2.32.5][203]
  * [2.32.4][204]
  * [2.32.3][205]
  * [2.32.2][206]
  * [2.32.1][207]
  * [2.32.0][208]
  * [2.31.8][209]
  * [2.31.7][210]
  * [2.31.6][211]
  * [2.31.5][212]
  * [2.31.4][213]
  * [2.31.3][214]
  * [2.31.2][215]
  * [2.31.1][216]
  * [2.31.0][217]
  * [2.30.0][218]
  * [2.29.1][219]
  * [2.29.0][220]
  * [2.28.0][221]
  * [2.27.0][222]
  * [2.26.0][223]
  * [2.25.0][224]
  * [2.24.1][225]
  * [2.24.0][226]
  * [2.23.1][227]
  * [2.23.0][228]
  * [2.22.0][229]
  * [2.21.0][230]
  * [2.20.1][231]
  * [2.20.0][232]
  * [2.19.1][233]
  * [2.19.0][234]
  * [2.18.4][235]
  * [2.18.3][236]
  * [2.18.2][237]
  * [2.18.1][238]
  * [2.18.0][239]
  * [2.17.0][240]
  * [2.16.1][241]
  * [2.16.0][242]
  * [2.15.1][243]
  * [2.15.0][244]
  * [2.14.2][245]
  * [2.14.1][246]
  * [2.14.0][247]
  * [2.13.2][248]
  * [2.13.1][249]
  * [2.13.0][250]
  * [2.12.1][251]
  * [2.12.0][252]
  * [2.11.1][253]
  * [2.11.0][254]
  * [2.10.2][255]
  * [2.10.1][256]
  * [2.10.0][257]
  * [2.9.0][258]
  * [2.8.6][259]
  * [2.8.5][260]
  * [2.8.4][261]
  * [2.8.3][262]
  * [2.8.2][263]
  * [2.8.1][264]
  * [2.8.0][265]
  * [2.7.3][266]
  * [2.7.2][267]
  * [2.7.1][268]
  * [2.7.0][269]
  * [2.6.1][270]
  * [2.6.0][271]
  * [2.5.1][272]
  * [2.5.0][273]
  * [2.4.3][274]
  * [2.4.2][275]
  * [2.4.1][276]
  * [2.4.0][277]
  * [2.3.0][278]
  * [2.2.3][279]
  * [2.2.2][280]
  * [2.2.1][281]
  * [2.2.0][282]
  * [2.1.0][283]
  * [2.0.0][284]
  * [1.6.0][285]
  * [1.5.3][286]
  * [1.5.2][287]
  * [1.5.1][288]
  * [1.5.0][289]
  * [1.4.4][290]
  * [1.4.3][291]
  * [1.4.2][292]
  * [1.4.1][293]
  * [1.4.0][294]
  * [1.3.2][295]
  * [1.3.1][296]
  * [1.3.0][297]
  * [1.2.3][298]
  * [1.2.2][299]
  * [1.2.1][300]
  * [1.2][301]
  * [1.1.1][302]
  * [1.1][303]
  * [1.0.3][304]
  * [1.0.2][305]
  * [1.0, 1.0.1][306]
  * [0.14.6][307]
  * [0.14.0][308]
  * [0.13.5][309]
  * [0.13.4][310]
  * [0.13.3][311]
  * [0.13.2][312]
  * [0.13.1][313]
  * [0.13.0][314]
  * [0.12.2][315]
  * [0.12.1][316]
  * [0.10.0][317]
  * [0.9.0][318]
  * [0.8.1][319]
  * [0.8][320]
  * [0.7.5][321]
  * [0.7.4][322]
  * [0.7.3][323]
  * [0.7.2][324]
  * [0.7.1][325]
  * [0.7][326]
  * [0.6.2][327]
  * [0.6.1][328]
  * [0.6][329]
  * [0.5.1][330]
  * [0.5][331]
  * [0.4][332]
[Next ][333]

© Copyright 2009-2025, Carl Byström, Jonatan Heyman, Lars Holmberg.

Built with [Sphinx][334] using a [theme][335] provided by [Read the Docs][336].

[1]: #locust-documentation
[2]: #getting-started
[3]: what-is-locust.html
[4]: what-is-locust.html#features
[5]: what-is-locust.html#name-background
[6]: what-is-locust.html#authors
[7]: what-is-locust.html#license
[8]: installation.html
[9]: installation.html#using-uvx-alternative
[10]: installation.html#done
[11]: installation.html#pre-release-builds
[12]: installation.html#install-for-development
[13]: installation.html#troubleshooting-installation
[14]: quickstart.html
[15]: quickstart.html#locust-s-web-interface
[16]: quickstart.html#direct-command-line-usage-headless
[17]: quickstart.html#more-options
[18]: quickstart.html#next-steps
[19]: #writing-locust-tests
[20]: writing-a-locustfile.html
[21]: writing-a-locustfile.html#auto-generating-a-locustfile
[22]: writing-a-locustfile.html#user-class
[23]: writing-a-locustfile.html#tasks
[24]: writing-a-locustfile.html#events
[25]: writing-a-locustfile.html#httpuser-class
[26]: writing-a-locustfile.html#tasksets
[27]: writing-a-locustfile.html#examples
[28]: writing-a-locustfile.html#how-to-structure-your-test-code
[29]: #running-your-locust-tests
[30]: configuration.html
[31]: increasing-request-rate.html
[32]: running-distributed.html
[33]: running-in-debugger.html
[34]: running-in-docker.html
[35]: running-without-web-ui.html
[36]: #running-with-locust-cloud
[37]: locust-cloud/locust-cloud.html
[38]: locust-cloud/docs.html
[39]: locust-cloud/docs.html#first-run
[40]: locust-cloud/docs.html#locustfile-and-mock-server
[41]: locust-cloud/docs.html#passing-options-to-locust
[42]: locust-cloud/docs.html#automated-runs-ci
[43]: locust-cloud/docs.html#extra-python-packages-and-files
[44]: locust-cloud/docs.html#view-dashboard-previous-test-runs
[45]: locust-cloud/docs.html#source-ip-addresses
[46]: locust-cloud/locust-cloud.html#kubernetes-operator
[47]: locust-cloud/kubernetes-operator.html
[48]: locust-cloud/kubernetes-operator.html#helm-charts
[49]: locust-cloud/kubernetes-operator.html#manifest-files
[50]: locust-cloud/kubernetes-operator.html#quickstart
[51]: locust-cloud/kubernetes-operator.html#locusttest-crd-configuration
[52]: locust-cloud/kubernetes-operator.html#general
[53]: locust-cloud/kubernetes-operator.html#locustfile-source
[54]: locust-cloud/kubernetes-operator.html#metadata
[55]: locust-cloud/kubernetes-operator.html#per-role-overrides
[56]: locust-cloud/kubernetes-operator.html#extended
[57]: locust-cloud/kubernetes-operator.html#examples
[58]: locust-cloud/kubernetes-operator.html#inline-locustfile
[59]: locust-cloud/kubernetes-operator.html#external-configmap-locustfile
[60]: locust-cloud/kubernetes-operator.html#custom-master-worker-pod-configuration
[61]: locust-cloud/kubernetes-operator.html#headless-run
[62]: locust-cloud/kubernetes-operator.html#upgrade
[63]: locust-cloud/kubernetes-operator.html#id1
[64]: locust-cloud/kubernetes-operator.html#uninstall
[65]: locust-cloud/kubernetes-operator.html#id2
[66]: locust-cloud/kubernetes-operator.html#id3
[67]: mailto:support%40locust.cloud
[68]: #other-functionalities
[69]: custom-load-shape.html
[70]: custom-load-shape.html#example
[71]: custom-load-shape.html#combining-users-with-different-load-profiles
[72]: custom-load-shape.html#restricting-which-user-types-to-spawn-in-each-tick
[73]: custom-load-shape.html#reusing-common-options-in-custom-shapes
[74]: retrieving-stats.html
[75]: testing-other-systems.html
[76]: testing-other-systems.html#xml-rpc
[77]: testing-other-systems.html#grpc
[78]: testing-other-systems.html#requests-based-libraries-sdks
[79]: testing-other-systems.html#rest
[80]: testing-other-systems.html#socketio
[81]: testing-other-systems.html#pytest
[82]: testing-other-systems.html#openai
[83]: testing-other-systems.html#mqtt
[84]: testing-other-systems.html#other-examples
[85]: increase-performance.html
[86]: increase-performance.html#how-to-use-fasthttpuser
[87]: increase-performance.html#concurrency
[88]: increase-performance.html#rest
[89]: increase-performance.html#connection-handling
[90]: increase-performance.html#api
[91]: extending-locust.html
[92]: extending-locust.html#request-context
[93]: extending-locust.html#adding-web-routes
[94]: extending-locust.html#extending-web-ui
[95]: extending-locust.html#adding-authentication-to-the-web-ui
[96]: extending-locust.html#run-a-background-greenlet
[97]: extending-locust.html#parametrizing-locustfiles
[98]: extending-locust.html#test-data-management
[99]: extending-locust.html#more-examples
[100]: logging.html
[101]: logging.html#options
[102]: logging.html#locust-loggers
[103]: use-as-lib.html
[104]: use-as-lib.html#skipping-monkey-patching
[105]: use-as-lib.html#full-example
[106]: telemetry.html
[107]: telemetry.html#setup
[108]: telemetry.html#exporters
[109]: telemetry.html#auto-instrumentation
[110]: telemetry.html#example
[111]: extensions.html
[112]: extensions.html#support-for-load-testing-other-protocols-reporting-etc
[113]: extensions.html#report-otel-traces-for-requests
[114]: extensions.html#automatically-translate-a-browser-recording-har-file-to-a-locustfile
[115]: extensions.html#workers-written-in-other-languages-than-python
[116]: vscode-extension.html
[117]: #further-reading-knowledgebase
[118]: developing-locust.html
[119]: further-reading.html
[120]: faq.html
[121]: #api
[122]: api.html
[123]: api.html#user-class
[124]: api.html#httpuser-class
[125]: api.html#httpsession-class
[126]: api.html#fasthttpuser-class
[127]: api.html#mqttuser-class
[128]: api.html#socketiouser-class
[129]: api.html#fasthttpsession-class
[130]: api.html#postgresuser-class
[131]: api.html#mongodbuser-class
[132]: api.html#milvususer-class
[133]: api.html#dnsuser-class
[134]: api.html#taskset-class
[135]: api.html#task-decorator
[136]: api.html#tag-decorator
[137]: api.html#sequentialtaskset-class
[138]: api.html#module-locust.wait_time
[139]: api.html#response-class
[140]: api.html#responsecontextmanager-class
[141]: api.html#exceptions
[142]: api.html#environment-class
[143]: api.html#event-hooks
[144]: api.html#runner-classes
[145]: api.html#web-ui-class
[146]: api.html#other
[147]: #changelog
[148]: changelog.html
[149]: changelog.html#id1
[150]: changelog.html#id2
[151]: changelog.html#id3
[152]: changelog.html#id4
[153]: changelog.html#id5
[154]: changelog.html#id6
[155]: changelog.html#id7
[156]: changelog.html#id8
[157]: changelog.html#id9
[158]: changelog.html#id10
[159]: changelog.html#id11
[160]: changelog.html#id12
[161]: changelog.html#id13
[162]: changelog.html#id14
[163]: changelog.html#id15
[164]: changelog.html#id16
[165]: changelog.html#id17
[166]: changelog.html#id18
[167]: changelog.html#id19
[168]: changelog.html#id20
[169]: changelog.html#id21
[170]: changelog.html#id22
[171]: changelog.html#id23
[172]: changelog.html#id24
[173]: changelog.html#id25
[174]: changelog.html#id26
[175]: changelog.html#id27
[176]: changelog.html#id28
[177]: changelog.html#id29
[178]: changelog.html#id30
[179]: changelog.html#id31
[180]: changelog.html#id32
[181]: changelog.html#id33
[182]: changelog.html#id34
[183]: changelog.html#id35
[184]: changelog.html#id36
[185]: changelog.html#id37
[186]: changelog.html#id38
[187]: changelog.html#id39
[188]: changelog.html#id40
[189]: changelog.html#id41
[190]: changelog.html#id42
[191]: changelog.html#id43
[192]: changelog.html#id44
[193]: changelog.html#id45
[194]: changelog.html#id46
[195]: changelog.html#id47
[196]: changelog.html#id48
[197]: changelog.html#id49
[198]: changelog.html#id50
[199]: changelog.html#id51
[200]: changelog.html#id52
[201]: changelog.html#id53
[202]: changelog.html#id54
[203]: changelog.html#id55
[204]: changelog.html#id56
[205]: changelog.html#id57
[206]: changelog.html#id58
[207]: changelog.html#id59
[208]: changelog.html#id60
[209]: changelog.html#id61
[210]: changelog.html#id62
[211]: changelog.html#id63
[212]: changelog.html#id64
[213]: changelog.html#id65
[214]: changelog.html#id66
[215]: changelog.html#id67
[216]: changelog.html#id68
[217]: changelog.html#id69
[218]: changelog.html#id70
[219]: changelog.html#id71
[220]: changelog.html#id72
[221]: changelog.html#id73
[222]: changelog.html#id74
[223]: changelog.html#id75
[224]: changelog.html#id76
[225]: changelog.html#id77
[226]: changelog.html#id78
[227]: changelog.html#id79
[228]: changelog.html#id80
[229]: changelog.html#id81
[230]: changelog.html#id82
[231]: changelog.html#id83
[232]: changelog.html#id84
[233]: changelog.html#id85
[234]: changelog.html#id86
[235]: changelog.html#id87
[236]: changelog.html#id88
[237]: changelog.html#id89
[238]: changelog.html#id90
[239]: changelog.html#id91
[240]: changelog.html#id92
[241]: changelog.html#id93
[242]: changelog.html#id94
[243]: changelog.html#id95
[244]: changelog.html#id96
[245]: changelog.html#id97
[246]: changelog.html#id98
[247]: changelog.html#id99
[248]: changelog.html#id100
[249]: changelog.html#id101
[250]: changelog.html#id102
[251]: changelog.html#id103
[252]: changelog.html#id104
[253]: changelog.html#id105
[254]: changelog.html#id106
[255]: changelog.html#id107
[256]: changelog.html#id108
[257]: changelog.html#id109
[258]: changelog.html#id110
[259]: changelog.html#id111
[260]: changelog.html#id112
[261]: changelog.html#id113
[262]: changelog.html#id114
[263]: changelog.html#id115
[264]: changelog.html#id116
[265]: changelog.html#id117
[266]: changelog.html#id118
[267]: changelog.html#id119
[268]: changelog.html#id120
[269]: changelog.html#id121
[270]: changelog.html#id122
[271]: changelog.html#id123
[272]: changelog.html#id124
[273]: changelog.html#id125
[274]: changelog.html#id126
[275]: changelog.html#id127
[276]: changelog.html#id128
[277]: changelog.html#id129
[278]: changelog.html#id130
[279]: changelog.html#id131
[280]: changelog.html#id132
[281]: changelog.html#id133
[282]: changelog.html#id134
[283]: changelog.html#id135
[284]: changelog.html#id136
[285]: changelog.html#id137
[286]: changelog.html#id138
[287]: changelog.html#id139
[288]: changelog.html#id140
[289]: changelog.html#id141
[290]: changelog.html#id142
[291]: changelog.html#id143
[292]: changelog.html#id144
[293]: changelog.html#id145
[294]: changelog.html#id146
[295]: changelog.html#id147
[296]: changelog.html#id148
[297]: changelog.html#id149
[298]: changelog.html#id150
[299]: changelog.html#id151
[300]: changelog.html#id152
[301]: changelog.html#id153
[302]: changelog.html#id154
[303]: changelog.html#id155
[304]: changelog.html#id156
[305]: changelog.html#id157
[306]: changelog.html#changelog-1-0
[307]: changelog.html#id159
[308]: changelog.html#id160
[309]: changelog.html#id161
[310]: changelog.html#id162
[311]: changelog.html#id163
[312]: changelog.html#id164
[313]: changelog.html#id165
[314]: changelog.html#id166
[315]: changelog.html#id167
[316]: changelog.html#id168
[317]: changelog.html#id169
[318]: changelog.html#id170
[319]: changelog.html#id171
[320]: changelog.html#id172
[321]: changelog.html#id173
[322]: changelog.html#id174
[323]: changelog.html#id175
[324]: changelog.html#id176
[325]: changelog.html#id177
[326]: changelog.html#id178
[327]: changelog.html#id179
[328]: changelog.html#id180
[329]: changelog.html#id181
[330]: changelog.html#id182
[331]: changelog.html#id183
[332]: changelog.html#id184
[333]: what-is-locust.html
[334]: https://www.sphinx-doc.org/
[335]: https://github.com/readthedocs/sphinx_rtd_theme
[336]: https://readthedocs.org
