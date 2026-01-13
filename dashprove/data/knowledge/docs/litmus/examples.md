# Chaos-Charts

[[Slack Channel]][1] [[GitHub Workflow]][2] [[Docker Pulls]][3] [[GitHub issues]][4] [[Twitter
Follow]][5] [[YouTube Channel]][6]


This repository hosts the Litmus Chaos Charts. A set of related chaos faults are bundled into a
Chaos Chart. Chaos Charts are classified into the following categories.

* [Kubernetes Chaos][7]
* [Application Chaos][8]
* [Platform Chaos][9]

### Kubernetes Chaos

Chaos faults that apply to Kubernetes resources are classified in this category. Following chaos
faults are supported for Kubernetes:

──────────────────────┬────────────────────────────────────────────────┬────────────────────────────
Fault Name            │Description                                     │Link                        
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Container Kill        │Kill one container in the application pod       │[ container-kill ][10]      
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Disk Fill             │Fill the Ephemeral Storage of the Pod           │[ disk-fill ][11]           
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Docker Service Kill   │Kill docker service of the target node          │[ docker-service-kill ][12] 
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Kubelet Service Kill  │Kill kubelet service of the target node         │[ kubelet-service-kill ][13]
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Node CPU Hog          │Stress the cpu of the target node               │[ node-cpu-hog ][14]        
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Node Drain            │Drain the target node                           │[ node-drain ][15]          
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Node IO Stress        │Stress the IO of the target node                │[ node-io-stress ][16]      
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Node Memory Hog       │Stress the memory of the target node            │[ node-memory-hog ][17]     
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Node Restart          │Restart the target node                         │[ node-restart ][18]        
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Node Taint            │Taint the target node                           │[ node-taint ][19]          
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod Autoscaler        │Scale the replicas of the target application    │[ pod-autoscaler ][20]      
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod CPU Hog           │Stress the CPU of the target pod                │[ pod-cpu-hog ][21]         
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod Delete            │Delete the target pods                          │[ pod-delete ][22]          
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod DNS Spoof         │Spoof dns requests to desired target hostnames  │[ pod-dns-spoof ][23]       
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod DNS Error         │Error the dns requests of the target pod        │[ pod-dns-error ][24]       
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod IO Stress         │Stress the IO of the target pod                 │[ pod-io-stress ][25]       
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod Memory Hog        │Stress the memory of the target pod             │[ pod-memory-hog ][26]      
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod Network Latency   │Induce the network latency in target pod        │[ pod-network-latency ][27] 
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod Network Corruption│Induce the network packet corruption in target  │[ pod-network-corruption    
                      │pod                                             │][28]                       
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod Network           │Induce the network packet duplication in target │[ pod-network-duplication   
Duplication           │pod                                             │][29]                       
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod Network Loss      │Induce the network loss in target pod           │[ pod-network-loss ][30]    
──────────────────────┼────────────────────────────────────────────────┼────────────────────────────
Pod Network Partition │Disrupt network connectivity to kubernetes pods │[ pod-network-partition     
                      │                                                │][31]                       
──────────────────────┴────────────────────────────────────────────────┴────────────────────────────

### Application Chaos

While chaos faults under the Kubernetes category offer the ability to induce chaos into Kubernetes
resources, it is difficult to analyze and conclude if the induced chaos found a weakness in a given
application. The application specific chaos faults are built with some checks on *pre-conditions*
and some expected outcomes after the chaos injection. The result of the chaos faults is determined
by matching the outcome with the expected outcome.

───────────────────┬───────────────────────────────────────────┬─────────────────────────
Fault Category     │Description                                │Link                     
───────────────────┼───────────────────────────────────────────┼─────────────────────────
Spring Boot Faults │Injects faults in Spring Boot applications │[ Spring Boot Faults][32]
───────────────────┴───────────────────────────────────────────┴─────────────────────────

### Platform Chaos

Chaos faults that inject chaos into the platform and infrastructure resources are classified into
this category. Management of platform resources vary significantly from each other, Chaos Charts may
be maintained separately for each platform (For example: AWS, GCP, Azure, VMWare etc.)

Following chaos faults are classified in this category:

───────────────┬───────────────────────────────┬────────────────────
Fault Category │Description                    │Link                
───────────────┼───────────────────────────────┼────────────────────
AWS Faults     │AWS Platform specific chaos    │[ AWS Faults ][33]  
───────────────┼───────────────────────────────┼────────────────────
Azure Faults   │Azure Platform specific chaos  │[ Azure Faults ][34]
───────────────┼───────────────────────────────┼────────────────────
GCP Faults     │GCP Platform specific chaos    │[ GCP Faults ][35]  
───────────────┼───────────────────────────────┼────────────────────
VMWare Faults  │VMWare Platform specific chaos │[ VMWare Faults     
               │                               │][36]               
───────────────┴───────────────────────────────┴────────────────────

## Installation Steps for Chart Releases

*Note: Supported from release 3.0.0*

* To install the chaos faults from a specific chart for a given release, execute the following
  commands with the desired `<release_version>`, `<chart_name>` & `<namespace>`
## downloads and unzips the released source
tar -zxvf <(curl -sL https://github.com/litmuschaos/chaos-charts/archive/<release_version>.tar.gz)

## installs the chaosexperiment resources
find chaos-charts-<release_version> -name experiments.yaml | grep <chart-name> | xargs kubectl apply
 -n <namespace> -f

* For example, to install the *Kubernetes* fault chart bundle for release *3.0.0*, in the
  *sock-shop* namespace, run:
tar -zxvf <(curl -sL https://github.com/litmuschaos/chaos-charts/archive/3.0.0.tar.gz)
find chaos-charts-3.0.0 -name experiments.yaml | grep kubernetes | xargs kubectl apply -n sock-shop 
-f

* If you would like to install a specific fault, replace the `experiments.yaml` in the above command
  with the relative path of the fault manifest within the parent chart. For example, to install only
  the *pod-delete* fault, run:
find chaos-charts-3.0.0 -name fault.yaml | grep 'kubernetes/pod-delete' | xargs kubectl apply -n soc
k-shop -f

## License

[[FOSSA Status]][37]

[1]: https://slack.litmuschaos.io
[2]: https://github.com/litmuschaos/chaos-charts/actions/workflows/push.yml/badge.svg?branch=master
[3]: https://hub.docker.com/r/litmuschaos/go-runner
[4]: https://github.com/litmuschaos/chaos-charts/issues
[5]: https://twitter.com/LitmusChaos
[6]: https://www.youtube.com/channel/UCa57PMqmz_j0wnteRa9nCaw
[7]: #kubernetes-chaos
[8]: #application-chaos
[9]: #platform-chaos
[10]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/container-kill
[11]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/disk-fill
[12]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/docker-service-kill
[13]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/kubelet-service-kill
[14]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/node-cpu-hog
[15]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/node-drain
[16]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/node-io-stress
[17]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/node-memory-hog
[18]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/node-restart
[19]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/node-taint
[20]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-autoscaler
[21]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-cpu-hog
[22]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-delete
[23]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-dns-spoof
[24]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-dns-error
[25]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-io-stress
[26]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-memory-hog
[27]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-network-latency
[28]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-network-corrupti
on
[29]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-network-duplicat
ion
[30]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-network-loss
[31]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/kubernetes/pod-network-partitio
n
[32]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/spring-boot
[33]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/aws
[34]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/azure
[35]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/gcp
[36]: https://github.com/litmuschaos/chaos-charts/tree/master/faults/vmware
[37]: https://app.fossa.io/projects/git%2Bgithub.com%2Flitmuschaos%2Fchaos-charts?ref=badge_large
