# IREE[link][1]

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment^{[1][2]}) is an
[MLIR][3]-based end-to-end compiler and runtime that lowers Machine Learning (ML) models to a
unified IR that scales up to meet the needs of the datacenter and down to satisfy the constraints
and special considerations of mobile and edge deployments.

## Key features[link][4]

* **Ahead-of-time compilation**
  
  Scheduling and execution logic are compiled together
  
  [ Project architecture][5]
* **Support for advanced model features**
  
  Dynamic shapes, flow control, streaming, and more
  
  [ Importing from ML frameworks][6]
* **Designed for CPUs, GPUs, and other accelerators**
  
  First class support for many popular devices and APIs
  
  [ Deployment configurations][7]
* **Low overhead, pipelined execution**
  
  Efficient power and resource usage on server and edge devices
  
  [ Benchmarking][8]
* **Binary size as low as 30KB on embedded systems**
  
  [ Running on bare-metal][9]
* **Debugging and profiling support**
  
  [ Profiling with Tracy][10]

## Support matrix[link][11]

IREE supports importing from a variety of ML frameworks:

* JAX
* ONNX
* PyTorch
* TensorFlow and TensorFlow Lite

The IREE compiler tools run on Linux, Windows, and macOS and can generate efficient code for a
variety of runtime platforms:

* Linux
* Windows
* macOS
* Android
* iOS
* Bare metal
* WebAssembly (experimental)

and architectures:

* ARM
* x86
* RISC-V

Support for hardware accelerators and APIs is also included:

* Vulkan
* ROCm/HIP
* CUDA
* Metal (for Apple silicon devices)
* AMD AIE (experimental)
* WebGPU (experimental)

## Project architecture[link][12]

IREE adopts a *holistic* approach towards ML model compilation: the IR produced contains both the
*scheduling* logic, required to communicate data dependencies to low-level parallel pipelined
hardware/API like [Vulkan][13], and the *execution* logic, encoding dense computation on the
hardware in the form of hardware/API-specific binaries like [SPIR-V][14].

[IREE Architecture] [IREE Architecture]

## Workflow overview[link][15]

Using IREE involves the following general steps:

1. **Import your model**
   
   Develop your program using one of the [supported frameworks][16], then import into IREE
2. **Select your [deployment configuration][17]**
   
   Identify your target platform, accelerator(s), and other constraints
3. **Compile your model**
   
   Compile through IREE, picking settings based on your deployment configuration
4. **Run your model**
   
   Use IREE's runtime components to execute your compiled model

### Importing models from ML frameworks[link][18]

IREE supports importing models from a growing list of [ML frameworks][19] and model formats:

* [ JAX][20]
* [ ONNX][21]
* [ PyTorch][22]
* [ TensorFlow][23] and [ TensorFlow Lite][24]

### Selecting deployment configurations[link][25]

IREE provides a flexible set of tools for various [deployment scenarios][26]. Fully featured
environments can use IREE for dynamic model deployments taking advantage of multi-threaded hardware,
while embedded systems can bypass IREE's runtime entirely or interface with custom accelerators.

* What platforms are you targeting? Desktop? Mobile? An embedded system?
* What hardware should the bulk of your model run on? CPU? GPU?
* How fixed is your model itself? Can the weights be changed? Do you want to support loading
  different model architectures dynamically?

IREE supports the full set of these configurations using the same underlying technology.

### Compiling models[link][27]

Model compilation is performed ahead-of-time on a *host* machine for any combination of *targets*.
The compilation process converts from layers and operators used by high level frameworks down into
optimized native code and associated scheduling logic.

For example, compiling for [GPU execution][28] using Vulkan generates SPIR-V kernels and Vulkan API
calls. For [CPU execution][29], native code with static or dynamic linkage and the associated
function calls are generated.

### Running models[link][30]

IREE offers a low level C API, as well as several sets of [API bindings][31] for compiling and
running programs using various languages.

## Community[link][32]

IREE is a [sandbox-stage project][33] of [LF AI & Data Foundation][34] made possible thanks to a
growing community of developers.

See how IREE is used:

[ Community][35]

### Project news[link][36]

* 2025-04-02: [AMD submitted an IREE-based SDXL implementation to the MLPerf benchmark suite][37]
* 2024-05-23: [IREE joins the LF AI & Data Foundation as a sandbox-stage project][38]

### Communication channels[link][39]

* [GitHub issues][40]: Feature requests, bugs, and other work tracking
* [IREE Discord server][41]: Daily development discussions with the core team and collaborators
* (New) [iree-announce email list][42]: Announcements
* (New) [iree-technical-discussion email list][43]: General and low-priority discussion
* (Legacy) [iree-discuss email list][44]: Announcements, general and low-priority discussion

## Project operations[link][45]

### Developer documentation[link][46]

Interested in contributing to IREE? Check out our developer documentation:

[ Developers][47]

### Roadmap[link][48]

IREE uses [GitHub Issues][49] for most work planning. Some subprojects use both [GitHub
Projects][50] and [GitHub Milestones][51] to track progress.

1. Pronounced "eerie" and often styled with the .cls-1, .cls-2 { fill: #cbcbcb; } .cls-1, .cls-2,
   .cls-3, .cls-4, .cls-5, .cls-6 { stroke-width: 0px; } .cls-1, .cls-4, .cls-6 { fill-rule:
   evenodd; } .cls-3, .cls-6 { fill: #01062c; } .cls-4 { fill: #e20d44; } .cls-5 { fill: #fff; }
   emoji [↩][52]

[1]: #iree
[2]: #fn:1
[3]: https://mlir.llvm.org/
[4]: #key-features
[5]: #project-architecture
[6]: #importing-models-from-ml-frameworks
[7]: #selecting-deployment-configurations
[8]: developers/performance/benchmarking/
[9]: guides/deployment-configurations/bare-metal/
[10]: developers/performance/profiling-with-tracy/
[11]: #support-matrix
[12]: #project-architecture
[13]: https://www.khronos.org/vulkan/
[14]: https://www.khronos.org/spir/
[15]: #workflow-overview
[16]: guides/ml-frameworks/
[17]: guides/deployment-configurations/
[18]: #importing-models-from-ml-frameworks
[19]: guides/ml-frameworks/
[20]: guides/ml-frameworks/jax/
[21]: guides/ml-frameworks/onnx/
[22]: guides/ml-frameworks/pytorch/
[23]: guides/ml-frameworks/tensorflow/
[24]: guides/ml-frameworks/tflite/
[25]: #selecting-deployment-configurations
[26]: guides/deployment-configurations/
[27]: #compiling-models
[28]: guides/deployment-configurations/gpu-vulkan/
[29]: guides/deployment-configurations/cpu/
[30]: #running-models
[31]: reference/bindings/
[32]: #community
[33]: https://lfaidata.foundation/projects/iree/
[34]: https://lfaidata.foundation/
[35]: community/
[36]: #project-news
[37]: https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.
html#stable-diffusion-xl-sdxl-text-to-image-mlperf-inference-benchmark
[38]: https://lfaidata.foundation/blog/2024/05/23/announcing-iree-a-new-initiative-for-machine-learn
ing-deployment/
[39]: #communication-channels
[40]: https://github.com/iree-org/iree/issues
[41]: https://discord.gg/wEWh6Z9nMU
[42]: https://lists.lfaidata.foundation/g/iree-announce
[43]: https://lists.lfaidata.foundation/g/iree-technical-discussion
[44]: https://groups.google.com/forum/#!forum/iree-discuss
[45]: #project-operations
[46]: #developer-documentation
[47]: developers/
[48]: #roadmap
[49]: https://github.com/iree-org/iree/issues
[50]: https://github.com/iree-org/iree/projects
[51]: https://github.com/iree-org/iree/milestones
[52]: #fnref:1
