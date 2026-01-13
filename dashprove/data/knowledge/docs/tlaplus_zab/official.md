## ZooKeeper: Because Coordinating Distributed Systems is a Zoo

ZooKeeper is a high-performance coordination service for distributed applications. It exposes common
services - such as naming, configuration management, synchronization, and group services - in a
simple interface so you don't have to write them from scratch. You can use it off-the-shelf to
implement consensus, group management, leader election, and presence protocols. And you can build on
it for your own, specific needs.

The following documents describe concepts and procedures to get you started using ZooKeeper. If you
have more questions, please ask the [mailing list][1] or browse the archives.

* **ZooKeeper Overview** Technical Overview Documents for Client Developers, Administrators, and
  Contributors
  
  * [Overview][2] - a bird's eye view of ZooKeeper, including design concepts and architecture
  * [Getting Started][3] - a tutorial-style guide for developers to install, run, and program to
    ZooKeeper
  * [Release Notes][4] - new developer and user facing features, improvements, and incompatibilities
* **Developers** Documents for Developers using the ZooKeeper Client API
  
  * [API Docs][5] - the technical reference to ZooKeeper Client APIs
  * [Programmer's Guide][6] - a client application developer's guide to ZooKeeper
  * [ZooKeeper Use Cases][7] - a series of use cases using the ZooKeeper.
  * [ZooKeeper Java Example][8] - a simple Zookeeper client application, written in Java
  * [Barrier and Queue Tutorial][9] - sample implementations of barriers and queues
  * [ZooKeeper Recipes][10] - higher level solutions to common problems in distributed applications
* **Administrators & Operators** Documents for Administrators and Operations Engineers of ZooKeeper
  Deployments
  
  * [Administrator's Guide][11] - a guide for system administrators and anyone else who might deploy
    ZooKeeper
  * [Quota Guide][12] - a guide for system administrators on Quotas in ZooKeeper.
  * [Snapshot and Restore Guide][13] - a guide for system administrators on take snapshot and
    restore ZooKeeper.
  * [JMX][14] - how to enable JMX in ZooKeeper
  * [Hierarchical Quorums][15] - a guide on how to use hierarchical quorums
  * [Oracle Quorum][16] - the introduction to Oracle Quorum increases the availability of a cluster
    of 2 ZooKeeper instances with a failure detector.
  * [Observers][17] - non-voting ensemble members that easily improve ZooKeeper's scalability
  * [Dynamic Reconfiguration][18] - a guide on how to use dynamic reconfiguration in ZooKeeper
  * [ZooKeeper CLI][19] - a guide on how to use the ZooKeeper command line interface
  * [ZooKeeper Tools][20] - a guide on how to use a series of tools for ZooKeeper
  * [ZooKeeper Monitor][21] - a guide on how to monitor the ZooKeeper
  * [Audit Logging][22] - a guide on how to configure audit logs in ZooKeeper Server and what
    contents are logged.
* **Contributors** Documents for Developers Contributing to the ZooKeeper Open Source Project
  
  * [ZooKeeper Internals][23] - assorted topics on the inner workings of ZooKeeper
* **Miscellaneous ZooKeeper Documentation**
  
  * [Wiki][24]
  * [FAQ][25]

[1]: http://zookeeper.apache.org/mailing_lists.html
[2]: zookeeperOver.html
[3]: zookeeperStarted.html
[4]: releasenotes.html
[5]: apidocs/zookeeper-server/index.html
[6]: zookeeperProgrammers.html
[7]: zookeeperUseCases.html
[8]: javaExample.html
[9]: zookeeperTutorial.html
[10]: recipes.html
[11]: zookeeperAdmin.html
[12]: zookeeperQuotas.html
[13]: zookeeperSnapshotAndRestore.html
[14]: zookeeperJMX.html
[15]: zookeeperHierarchicalQuorums.html
[16]: zookeeperOracleQuorums.html
[17]: zookeeperObservers.html
[18]: zookeeperReconfig.html
[19]: zookeeperCLI.html
[20]: zookeeperTools.html
[21]: zookeeperMonitor.html
[22]: zookeeperAuditLogs.html
[23]: zookeeperInternals.html
[24]: https://cwiki.apache.org/confluence/display/ZOOKEEPER
[25]: https://cwiki.apache.org/confluence/display/ZOOKEEPER/FAQ
