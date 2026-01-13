* Getting Started
* Installation and Deployment
* Quick Start
Version: 2.8.0
On this page

# Quick Start

This document describes how to start Chaos Mesh quickly in a test or local environment.

caution

**In this document, Chaos Mesh is installed via a script for quick trial only.**

If you need to install Chaos Mesh in the production environment or other strict non-test scenarios,
it is recommended to use [Helm][1]. For more details, refer to [Installation using Helm][2].

## Environment preparation[​][3]

Please ensure that the Kubernetes cluster is deployed in the environment before the trial. If the
Kubernetes cluster has not been deployed, you can refer to the links below to complete the
deployment:

* [Kubernetes][4]
* [minikube][5]
* [kind][6]
* [K3s][7]
* [Microk8s][8]

## Quick installation[​][9]

To install Chaos Mesh in a test environment, run the following script:

* K8s
* kind
* K3s
* MicroK8s

If you want to specify a `kind` version, add the `--kind-version xx` parameter at the end of the
script, for example:

After running, Chaos Mesh will automatically install the appropriate version of
CustomResourceDefinitions and the required components.

For more installation details, refer to the source code of the [`install.sh`][10].

## Verify the installation[​][11]

To check the running status of Chaos Mesh, execute the following command:

`kubectl get pods -n chaos-mesh -l app.kubernetes.io/instance=chaos-mesh
`

The expected output is as follows:

`NAME                                       READY   STATUS    RESTARTS   AGE
chaos-controller-manager-7b8c86cc9-44dzf   1/1     Running   0          17m
chaos-controller-manager-7b8c86cc9-mxw99   1/1     Running   0          17m
chaos-controller-manager-7b8c86cc9-xmc5v   1/1     Running   0          17m
chaos-daemon-sg2k2                         1/1     Running   0          17m
chaos-dashboard-b9dbc6b68-hln25            1/1     Running   0          17m
chaos-dns-server-546675d89d-qkjqq          1/1     Running   0          17m
`

If your actual output is similar to the expected output, then Chaos Mesh has been successfully
installed.

note

If the `STATUS` of your actual output is not `Running`, then execute the following command to check
the Pod details, and troubleshoot issues according to the error information.

`# Take the chaos-controller as an example
kubectl describe po -n chaos-mesh chaos-controller-manager-7b8c86cc9-44dzf
`
note

If leader election is turned off, `chaos-controller-manager` should only have 1 replication.

`NAME                                        READY   STATUS    RESTARTS   AGE
chaos-controller-manager-676d8567c7-ndr5j   1/1     Running   0          24m
chaos-daemon-6l55b                          1/1     Running   0          24m
chaos-dashboard-b9dbc6b68-hln25             1/1     Running   0          44m
chaos-dns-server-546675d89d-qkjqq           1/1     Running   0          44m
`

## Run Chaos experiments[​][12]

After verifying that the installation is complete, you can run a Chaos experiment to experience the
features of Chaos Mesh.

For the method to run the experiment, it is recommended to refer to [Run a Chaos experiment][13].
After successfully creating the experiment, you can observe the running status of the experiment on
the Chaos Dashboard.

## Uninstall Chaos Mesh[​][14]

To uninstall Chaos Mesh, execute the following command:

You can also delete the `chaos-mesh` namespace to directly uninstall Chaos Mesh:

`kubectl delete ns chaos-mesh
`

## FAQ[​][15]

### Why the `local` directory appears in the root directory after installation?[​][16]

If you don't install `kind` in the existing environment, and you use the `--local kind` parameter
when executing the installation command, the `install.sh` script will automatically install the
`kind` in the `local` directory under the root directory.

[Edit this page][17]

[1]: https://helm.sh/
[2]: /docs/production-installation-using-helm/
[3]: #environment-preparation
[4]: https://kubernetes.io/docs/setup/
[5]: https://minikube.sigs.k8s.io/docs/start/
[6]: https://kind.sigs.k8s.io/docs/user/quick-start/
[7]: https://rancher.com/docs/k3s/latest/en/quick-start/
[8]: https://microk8s.io/
[9]: #quick-installation
[10]: https://github.com/chaos-mesh/chaos-mesh/blob/master/install.sh
[11]: #verify-the-installation
[12]: #run-chaos-experiments
[13]: /docs/run-a-chaos-experiment/
[14]: #uninstall-chaos-mesh
[15]: #faq
[16]: #why-the-local-directory-appears-in-the-root-directory-after-installation
[17]: https://github.com/chaos-mesh/website/edit/master/versioned_docs/version-2.8.0/quick-start.md
