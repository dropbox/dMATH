Version: 3.23.0
On this page

# ChaosCenter installation

## Prerequisites[​][1]

* Kubernetes 1.17 or later
* A Persistent volume of 20GB
note

Recommend to have a Persistent volume(PV) of 20GB, You can start with 1GB for test purposes as well.
This PV is used as persistent storage to store the chaos config and chaos-metrics in the Portal. By
default, litmus install would use the default storage class to allocate the PV. Provide this value

* [Helm3][2] or [kubectl][3]

## Installation[​][4]

Users looking to use Litmus for the first time have two options available to them today. One way is
to use a hosted Litmus service like [Harness Chaos Engineering SaaS][5]. Alternatively, users
looking for some more flexibility can install Litmus into their own Kubernetes cluster.

Users choosing the self-hosted option can refer to our Install and Configure docs for installing
alternate versions and more detailed instructions.

* Self-Hosted
* Hosted (Beta)
Installation of Self-Hosted Litmus can be done using either of the below methods:

[Helm3][6] chart

[Kubectl][7] yaml spec file

Refer to the below details for Self-Hosted Litmus installation.
[Harness][8] offers a free service for community members which makes getting started with Litmus
easy. Create an account to get started. Once logged in, create a new hosted control plane and
connect to it via the up CLI. Litmus can be used as a hosted cloud service using [Harness Chaos
Engineering SaaS][9]. Harness Chaos Engineering SaaS executes your Chaos Experiments in the cloud by
managing all your Chaos Control Plane components, while the Chaos Execution Plane components exist
on your Kubernetes cluster as part of an external chaos infrastructure.

To get started with Harness Chaos Engineering SaaS, visit [Harness Chaos Engineering SaaS][10] and
register for free. You can skip the below installation steps.
note

With 3.9.0 release, Cluster scope installation is deprecated. Now Namespaced mode is the only
supported and standard installation mode.

Although Litmus will be deployed in a particular namespace, chaos infrastructure can be deployed in
both cluster as well as namespace mode. [Learn more here][11].

### Install Litmus using Helm[​][12]

The helm chart will install all the required service account configuration and ChaosCenter.

The following steps will help you install Litmus ChaosCenter via helm.

#### Step-1: Add the litmus helm repository[​][13]

`helm repo add litmuschaos https://litmuschaos.github.io/litmus-helm/
helm repo list
`
Copy

#### Step-2: Create the namespace on which you want to install Litmus ChaosCenter[​][14]

* The ChaosCenter can be placed in any namespace, but for this scenario we are choose `litmus` as
  the namespace.
`kubectl create ns litmus
`
Copy

#### Step-3: Install Litmus ChaosCenter[​][15]

`helm install chaos litmuschaos/litmus --namespace=litmus --set portal.frontend.service.type=NodePor
t
`
Copy

> **Note:** If your Kubernetes cluster isn't local, you may want not to expose Litmus via
> `NodePort`. If so, remove `--set portal.frontend.service.type=NodePort` option. To connect to
> Litmus UI from your laptop, you can use `kubectl port-forward svc/chaos-litmus-frontend-service
> 9091:9091`. Then you can use your browser and open `127.0.0.1:9091`.

* If your Kubernetes cluster is local (such as minikube or kind) and only accessing Litmus locally,
  please replace the default endpoint with your custom CHAOS_CENTER_UI_ENDPOINT as shown below.
  
  `helm install chaos litmuschaos/litmus --namespace=litmus \
  --set portal.frontend.service.type=NodePort \
  --set portal.server.graphqlServer.genericEnv.CHAOS_CENTER_UI_ENDPOINT=http://chaos-litmus-frontend
  -service.litmus.svc.cluster.local:9091
  `
  Copy
* Litmus helm chart depends on `bitnami/mongodb` [helm chart][16], which uses a mongodb image not
  supported on ARM. If you want to install Litmus on an ARM-based server, please replace the default
  one with your custom mongodb arm image as shown below.
  
  `helm install chaos litmuschaos/litmus --namespace=litmus \
  --set portal.frontend.service.type=NodePort \
  --set mongodb.image.registry=<put_registry> \
  --set mongodb.image.repository=<put_image_repository> \
  --set mongodb.image.tag=<put_image_tag>
  `
  Copy
Expected Output
`NAME: chaos
LAST DEPLOYED: Tue Jun 15 19:20:09 2021
NAMESPACE: litmus
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
Thank you for installing litmus 😀

Your release is named chaos and its installed to namespace: litmus.

Visit https://docs.litmuschaos.io to find more info.
`
Copy

> **Note:** Litmus uses Kubernetes CRDs to define chaos intent. Helm3 handles CRDs better than
> Helm2. Before you start running a chaos experiment, verify if Litmus is installed correctly.

## Install Litmus using kubectl[​][17]

In this method the users need to install mongo first via helm and then apply the installation
manifest. Follow the instructions [here][18].

### Install mongo[​][19]

` helm repo add bitnami https://charts.bitnami.com/bitnami
`
Copy

Mongo Values

`auth:
  enabled: true
  rootPassword: "1234"
  # -- existingSecret Existing secret with MongoDB(&reg;) credentials (keys: `mongodb-passwords`, `m
ongodb-root-password`, `mongodb-metrics-password`, ` mongodb-replica-set-key`)
  existingSecret: ""
architecture: replicaset
replicaCount: 3
persistence:
  enabled: true
volumePermissions:
  enabled: true
metrics:
  enabled: false
  prometheusRule:
    enabled: false

# bitnami/mongodb is not yet supported on ARM.
# Using unofficial tools to build bitnami/mongodb (arm64 support)
# more info: https://github.com/ZCube/bitnami-compat
#image:
#  registry: ghcr.io/zcube
#  repository: bitnami-compat/mongodb
#  tag: 6.0.5
`
Copy
`helm install my-release bitnami/mongodb --values mongo-values.yml -n <NAMESPACE> --create-namespace
`
Copy

Litmus supports for HTTP and HTTPS mode of installation.

### Basic installation (HTTP based and allows all origins)[​][20]

Applying the manifest file will install all the required service account configuration and
ChaosCenter in namespaced scope.

` kubectl apply -f https://raw.githubusercontent.com/litmuschaos/litmus/master/mkdocs/docs/3.20.0/li
tmus-getting-started.yaml -n <NAMESPACE>
`
Copy

### Advanced installation (HTTPS based and CORS rules apply)[​][21]

For advanced installation visit [here][22]

## Verify your installation[​][23]

#### Verify if the frontend, server, and database pods are running[​][24]

* Check the pods in the namespace where you installed Litmus:
  
  `kubectl get pods -n litmus
  `
  Copy
  Expected Output
  `NAME                                       READY   STATUS    RESTARTS   AGE
  litmusportal-server-6fd57cc89-6w5pn        1/1     Running     0          57s
  litmusportal-auth-server-7b596fff9-5s6g5   1/1     Running     0          57s
  litmusportal-frontend-55974fcf59-cxxrf     1/1     Running     0          58s
  my-release-mongodb-0                       1/1     Running     0          63s
  my-release-mongodb-1                       1/1     Running     0          63s
  my-release-mongodb-2                       1/1     Running     0          62s
  my-release-mongodb-arbiter-0               1/1     Running     0          64s
  
  `
  Copy
* Check the services running in the namespace where you installed Litmus:
  
  `kubectl get svc -n litmus
  `
  Copy
  Expected Output
  `NAME                                  TYPE           CLUSTER-IP     EXTERNAL-IP      PORT(S)     
                      AGE
  chaos-exporter                        ClusterIP      10.68.45.7     <none>           8080/TCP     
                     23h
  litmusportal-auth-server-service      NodePort       10.68.34.91    <none>           9003:32368/TC
  P,3030:31051/TCP   23h
  litmusportal-frontend-service         NodePort       10.68.43.68    <none>           9091:30070/TC
  P                  23h
  litmusportal-server-service           NodePort       10.68.33.242   <none>           9002:32455/TC
  P,8000:30722/TCP   23h
  my-release-mongodb-arbiter-headless   ClusterIP      None           <none>           27017/TCP    
                     23h
  my-release-mongodb-headless           ClusterIP      None           <none>           27017/TCP    
                     23h
  workflow-controller-metrics           ClusterIP      10.68.33.65    <none>           9090/TCP     
                     23h
  `
  Copy

## Accessing the ChaosCenter[​][25]

To setup and login to ChaosCenter expand the available services just created and copy the `PORT` of
the `litmusportal-frontend-service` service

`kubectl get svc -n litmus
`
Copy
Expected Output
`NAME                               TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)                
         AGE
litmusportal-frontend-service      NodePort    10.43.79.17    <none>        9091:31846/TCP          
        102s
litmusportal-server-service        NodePort    10.43.30.54    <none>        9002:31245/TCP,8000:3271
4/TCP   101s
litmusportal-auth-server-service   NodePort    10.43.81.108   <none>        9003:32618/TCP,3030:3189
9/TCP   101s
mongo-service                      ClusterIP   10.43.227.10   <none>        27017/TCP               
        101s
mongo-headless-service             ClusterIP   None           <none>        27017/TCP               
        101s
`
Copy

> **Note**: In this case, the PORT for `litmusportal-frontend-service` is `31846`. Yours will be
> different.

Once you have the PORT copied in your clipboard, simply use your IP and PORT in this manner
`<NODEIP>:<PORT>` to access the Litmus ChaosCenter.

For example:

`http://172.17.0.3:31846/
`
Copy

> Where `172.17.0.3` is my NodeIP and `31846` is the frontend service PORT. If using a LoadBalancer,
> the only change would be to provide a `<LoadBalancerIP>:<PORT>`. [Learn more about how to access
> ChaosCenter with LoadBalancer][26]

**NOTE:** With advanced installation CORS rules are applied, once manifest is applied frontend
loadbalancer IP needs to be added in the `ALLOWED_ORIGINS` environment in both auth and graphql
server deployment.

You should be able to see the Login Page of Litmus ChaosCenter. The **default credentials** are

`Username: admin
Password: litmus
`
Copy

By default you are assigned with a default project with Owner permissions.

## Learn more[​][27]

* [Install ChaosCenter with HTTPS][28]
* [Connect External Chaos Infrastructures to ChaosCenter][29]
* [Setup Endpoints and Access ChaosCenter without Ingress][30]
* [Setup Endpoints and Access ChaosCenter with Ingress][31]
[Edit this page][32]

[1]: #prerequisites
[2]: https://v3.helm.sh/
[3]: https://kubernetes.io/docs/tasks/tools/#kubectl
[4]: #installation
[5]: https://app.harness.io/auth/#/signin
[6]: #install-litmus-using-helm
[7]: #install-litmus-using-kubectl
[8]: https://harness.io/
[9]: https://app.harness.io/auth/#/signin
[10]: https://developer.harness.io/docs/chaos-engineering/get-started/learn-more-free-plan
[11]: https://docs.litmuschaos.io/docs/concepts/chaos-infrastructure
[12]: #install-litmus-using-helm
[13]: #step-1-add-the-litmus-helm-repository
[14]: #step-2-create-the-namespace-on-which-you-want-to-install-litmus-chaoscenter
[15]: #step-3-install-litmus-chaoscenter
[16]: https://github.com/bitnami/charts/tree/main/bitnami/mongodb
[17]: #install-litmus-using-kubectl
[18]: https://github.com/litmuschaos/litmus/tree/master/chaoscenter#installation-steps-for-litmus-30
0-beta9
[19]: #install-mongo
[20]: #basic-installation-http-based-and-allows-all-origins
[21]: #advanced-installation-https-based-and-cors-rules-apply
[22]: /docs/user-guides/chaoscenter-advanced-installation
[23]: #verify-your-installation
[24]: #verify-if-the-frontend-server-and-database-pods-are-running
[25]: #accessing-the-chaoscenter
[26]: /docs/user-guides/setup-without-ingress#with-loadbalancer
[27]: #learn-more
[28]: /docs/user-guides/chaoscenter-advanced-installation
[29]: /docs/user-guides/chaos-infrastructure-installation
[30]: /docs/user-guides/setup-without-ingress
[31]: /docs/user-guides/setup-with-ingress
[32]: https://github.com/litmuschaos/litmus-docs/edit/master/website/versioned_docs/version-3.23.0/g
etting-started/installation.md
