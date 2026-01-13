# NVIDIA TensorRT

NVIDIA® TensorRT™ is an ecosystem of tools for developers to achieve high-performance deep learning
inference. TensorRT includes inference compilers, runtimes, and model optimizations that deliver low
latency and high throughput for production applications. The TensorRT ecosystem includes the
TensorRT compiler, TensorRT-LLM, TensorRT Model Optimizer, TensorRT for RTX, and TensorRT Cloud.

[Download Now][1][Documentation
][2][GitHub][3]

## How TensorRT Works

Speed up inference by 36X compared to CPU-only platforms.

Built on the NVIDIA® CUDA® parallel programming model, TensorRT includes libraries that optimize
neural network models trained on all major frameworks, calibrate them for lower precision with high
accuracy, and deploy them to hyperscale data centers, workstations, laptops, and edge devices.
TensorRT optimizes inference using quantization, layer and tensor fusion, and kernel tuning
techniques.

NVIDIA TensorRT Model Optimizer provides easy-to-use quantization techniques, including
post-training quantization and quantization-aware training to compress your models. FP8, FP4, INT8,
INT4, and advanced techniques such as AWQ are supported for your deep learning inference
optimization needs. Quantized inference significantly minimizes latency and memory bandwidth, which
is required for many real-time services, autonomous and embedded applications.

### Read the Introductory TensorRT Blog

Learn how to apply TensorRT optimizations and deploy a PyTorch model to GPUs.


[Read Blog][4]

### Watch On-Demand TensorRT Sessions From GTC

Learn more about TensorRT and its features from a curated list of webinars at GTC.

[Watch Sessions][5]

### Get the Complete Developer Guide

See how to get started with TensorRT in this step-by-step developer and API reference guide.



[Read Guide][6]

### Navigate AI infrastructure and Performance

Learn how to lower your cost per token and get the most out of your AI models with our ebook.



[View Ebook][7]

## Key Features

### Large Language Model Inference

[NVIDIA TensorRT-LLM][8] is an open-source library that accelerates and optimizes inference
performance of large language models (LLMs) on the NVIDIA AI platform with a simplified Python API.
Developers accelerate LLM performance on NVIDIA GPUs in the data center or on workstation GPUs.

### Compile in the Cloud

NVIDIA TensorRT Cloud is a developer-focused service for generating hyper-optimized engines for
given constraints and KPIs. Given an LLM and inference throughput/latency requirements, a developer
can invoke TensorRT Cloud service using a command-line interface to hyper-optimize a TensorRT-LLM
engine for a target GPU. The cloud service will automatically determine the best engine
configuration that meets the requirements. Developers can also use the service to build optimized
TensorRT engines from ONNX models on a variety of NVIDIA RTX, GeForce, Quadro®, or Tesla®-class
GPUs.

TensorRT Cloud is available with limited access to select partners. [Apply][9] for access, subject
to approval.

### Optimize Neural Networks

[TensorRT Model Optimizer][10] is a unified library of state-of-the-art model optimization
techniques, including quantization, pruning, speculation, sparsity, and distillation. It compresses
deep learning models for downstream deployment frameworks like TensorRT-LLM, TensorRT, vLLM, and
SGLang to efficiently optimize inference on NVIDIA GPUs. TensorRT Model Optimizer also supports
training for inference techniques such as Speculative Decoding Module Training,
Pruning/Distillation, and Quantization Aware Training through NeMo and Hugging Face frameworks.

### Major Framework Integrations
### 

TensorRT integrates directly into [PyTorch][11] and [Hugging Face][12] to achieve 6X faster
inference with a single line of code. TensorRT provides an [ONNX][13] parser to import [ONNX][14]
models from popular frameworks into TensorRT. [MATLAB][15] is integrated with TensorRT through GPU
Coder to automatically generate high-performance inference engines for NVIDIA Jetson™, NVIDIA
DRIVE®, and data center platforms.

### Deploy, Run, and Scale With Dynamo-Triton

TensorRT-optimized models are deployed, run, and scaled with [NVIDIA Dynamo Triton][16]
inference-serving software that includes TensorRT as a backend. The advantages of using Triton
include high throughput with dynamic batching, concurrent model execution, model ensembling, and
streaming audio and video inputs.

### Simplify AI deployment on RTX

TensorRT for RTX offers an optimized inference deployment solution for NVIDIA RTX GPUs. It
facilitates faster engine build times within 15 to 30s, facilitating apps to build inference engines
directly on target RTX PCs during app installation or on first run, and does so within a total
library footprint of under 200 MB, minimizing memory footprint. Engines built with TensorRT for RTX
are cross-OS, cross-GPU portable, ensuring a build once, deploy anywhere workflow.

### Accelerate Every Inference Platform

TensorRT can optimize models for applications across the edge, laptops, desktops, and data centers.
It powers key NVIDIA solutions—such as NVIDIA TAO, NVIDIA DRIVE, NVIDIA Clara™, and NVIDIA
JetPack™—and is integrated with application-specific SDKs, such as NVIDIA NIM™, NVIDIA DeepStream,
NVIDIA Riva, NVIDIA Merlin™, NVIDIA Maxine™, NVIDIA Morpheus, and NVIDIA Broadcast Engine.

TensorRT provides developers a unified path to deploy intelligent video analytics, speech AI,
recommender systems, video conferencing, AI-based cybersecurity, and streaming apps in production.

## Get Started With TensorRT

TensorRT is an ecosystem of APIs for building and deploying high-performance deep learning
inference. It offers a variety of inference solutions for different developer requirements.

───────────────────────────────────┬───────────────────────┬────────────────────────────────────────
Use-case                           │Deployment Platform    │Solution                                
───────────────────────────────────┼───────────────────────┼────────────────────────────────────────
Inference for LLMs                 │Data center GPUs like  │Download TRT-LLM                        
                                   │GB100, H100, A100, etc.│                                        
                                   │                       │TensorRT-LLM is available for free on   
                                   │                       │[GitHub][17].                           
                                   │                       │                                        
                                   │                       │                                        
                                   │                       │[Download (GitHub)][18]                 
                                   │                       │                                        
                                   │                       │[Documentation][19]                     
───────────────────────────────────┼───────────────────────┼────────────────────────────────────────
Inference for non-LLMs like CNNs,  │Data center GPUs,      │Download TensorRT                       
Diffusions, Transformers, etc.     │Embedded, and Edge     │                                        
                                   │platforms              │The TensorRT inference library provides 
Safety-compliant and               │                       │a general-purpose AI compiler and an    
high-performance inference for     │                       │inference runtime that delivers low     
Automotive Embedded                │Automotive platform:   │latency and high throughput for         
                                   │NVIDIA DRIVE AGX       │production applications.                
Inference for non-LLMs in robotics │                       │                                        
and edge applications              │                       │                                        
                                   │Edge Platform: Jetson, │[Download SDK][20]                      
                                   │NVIDIA IGX, etc.       │                                        
                                   │                       │[Download Container][21]                
───────────────────────────────────┼───────────────────────┼────────────────────────────────────────
AI Model Inferencing on RTX PCs    │NVIDIA GeForce RTX and │Download TensorRT for RTX               
                                   │RTX Pro GPUs in laptops│                                        
                                   │and desktops           │TensorRT for RTX is a dedicated         
                                   │                       │inference deployment solution for RTX   
                                   │                       │GPUs.                                   
                                   │                       │                                        
                                   │                       │                                        
                                   │                       │[Download SDK][22]                      
                                   │                       │                                        
                                   │                       │[Documentation][23]                     
───────────────────────────────────┼───────────────────────┼────────────────────────────────────────
Model optimizations like           │Data center GPUs like  │Download TensorRT Model Optimizer       
Quantization, Distillation,        │GB100, H100, etc.      │                                        
Sparsity, etc.                     │                       │TensorRT Model Optimizer is free on     
                                   │                       │NVIDIA PyPI, with examples and recipes  
                                   │                       │on [GitHub][24].                        
                                   │                       │                                        
                                   │                       │                                        
                                   │                       │[Download (GitHub)][25]                 
                                   │                       │                                        
                                   │                       │[Documentation][26]                     
───────────────────────────────────┴───────────────────────┴────────────────────────────────────────

## Get Started With TensorRT Frameworks

TensorRT Frameworks add TensorRT compiler functionality to frameworks like PyTorch.

[TensorRT speeds up inference by 36X]

### Download ONNX and Torch-TensorRT

The TensorRT inference library provides a general-purpose AI compiler and an inference runtime that
delivers low latency and high throughput for production applications.

ONYX:

[Documentation][27]

Torch-TensorRT:

[Download Container][28]
[Documentation][29]
[TensorRT speeds up inference by 36X]

### Experience Tripy: Pythonic Inference With TensorRT

Experience high-performance inference and excellent usability with Tripy. Expect intuitive APIs,
easy debugging with eager mode, clear error messages, and top-notch documentation to streamline your
deep learning deployment.

[Documentation][30]
[Examples][31]
[Contribute][32]
[TensorRT speeds up inference by 36X]

### Deploy

Get a free license to try [NVIDIA AI Enterprise][33] in production for 90 days using your existing
infrastructure.

[Request a 90-Day License][34]

## World-Leading Inference Performance

TensorRT was behind NVIDIA’s wins across all [inference performance][35] tests in the
industry-standard benchmark for [MLPerf Inference][36]. TensorRT-LLM accelerates the latest large
language models for [generative AI][37], delivering up to 8X more performance, 5.3X better TCO, and
nearly 6X lower energy consumption.

[See All Benchmarks][38]

### 8X Increase in GPT-J 6B Inference Performance

[TensorRT-LLM on H100 has 8X increase in GPT-J 6B inference performance]

### 4X Higher Llama2 Inference Performance

[TensorRT-LLM on H100 has 4X Higher Llama2 Inference Performance]

### Total Cost of Ownership

Lower is better
[TensorRT-LLM has lower total cost of ownership than GPT-J 6B and Llama 2 70B]

### Energy Use

Lower is better
[TensorRT-LLM has lower energy use than GPT-J 6B and Llama 2 70B]

#### NVIDIA Blackwell Delivers Unmatched Performance and ROI for AI Inference

The NVIDIA Blackwell platform—including NVFP4 low precision format, fifth-generation NVIDIA NVLink
and NVLink Switch, and the NVIDIA TensorRT-LLM and NVIDIA Dynamo inference frameworks—enables the
highest AI factory revenue: A $5M investment in GB200 NVL72 generates $75 million in token revenue—a
15x return on investment. This includes development with community frameworks such as SGLang, vLLM,
and more.

[Explore technical results][39]
[NVIDIA Rivermax provides real-time streaming for the Las Vegas Sphere, world’s largest LED display]

## Starter Kits

### Beginner Guide to TensorRT

* [View Quick-Start Guide][40]
* [View Quick-Start Notebooks][41]
* Read Blog: [Speeding Up Deep Learning Inference Using NVIDIA TensorRT][42]
* Read Blog: [Optimizing and Serving Models With TensorRT and Triton][43]
* Watch Video: [Getting Started With NVIDIA TensorRT][44]

### Beginner Guide to TensorRT-LLM

* [View Quick-Start Guide][45]
* [View Quick-Start Notebooks][46]
* Read Blog: [Speeding Up Deep Learning Inference Using NVIDIA TensorRT][47]
* Read Blog: [Optimizing and Serving Models With TensorRT and Triton][48]
* Watch Video: [Getting Started With NVIDIA TensorRT][49]

### Beginner Guide to TensorRT Model Optimizer

* [Reference Architecture][50]
* [Workflow Guide & Documentation][51]
* [Training Courses][52]
* [NVIDIA Omniverse Blueprint for Precise Visual Generative AI][53]

### Beginner Guide to Torch-TensorRT

* Watch Video: [Getting Started With NVIDIA Torch-TensorRT][54]
* Read Blog: [Accelerate Inference up to 6X in PyTorch][55]
* Download Notebook: [Object Detection With SSD][56] (Jupyter Notebook)

### Beginner Guide to TensorRT Pythonic Frontend: Tripy

* [Introduction Guide][57]
* [ResNet-50 notebook][58]
* [nanoGPT][59]
* [Segment Anything Model V2][60]

### Beginner Guide to TensorRT for RTX

* [View Quick Start Guide
  ][61]
* [Access Samples and Demos][62]
* [Read Blog: ][63]
  
  # [Run High-Performance AI Applications with NVIDIA TensorRT for RTX][64]
  
* [Access TensorRT for RTX through WindowsML
  ][65]

## TensorRT Learning Library

OSS (Github)

Quantization Quickstart

NVIDIA TensorRT-LLM

The [PyTorch backend][66] supports FP8 and NVFP4 quantization. Explore [GitHub][67] to pass
quantized models in the Hugging Face model hub, which are generated by TensorRT Model Optimizer.

[Link to GitHub][68]
[Link to PyTorch Documentation][69]

OSS (Github)

Adding a New Model in PyTorch Backend

This guide provides a step-by-step process for adding a new model in PyTorch Backend.

[Link to GitHub][70]

OSS (Github)

Using TensoRT-Model Optimizer for Speculative Decoding

ModelOpt’s Speculative Decoding module enables your model to generate multiple tokens in each
generation step. This can be useful for reducing the latency of your model and speeding up
inference.

[Link to GitHub][71]

## TensorRT Ecosystem Ecosystem

Widely Adopted Across Industries

[NVIDIA TensorRT is widely adopted by top companies across industries]

## More Resources

[NVIDIA Developer Forums]

### Explore the Community

[NVIDIA Training and Certification]

### Get Training and Certification

[NVIDIA Inception Program for Startups]

### Read Top Stories and Blogs

## Ethical AI

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and
practices to enable development for a wide array of AI applications. When downloaded or used in
accordance with our terms of service, developers should work with their supporting model team to
ensure this model meets requirements for the relevant industry and use case and addresses unforeseen
product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++
Explainability, Bias, Safety & Security, and Privacy Subcards. Please report security
vulnerabilities or NVIDIA AI Concerns [here][72].

Get started with TensorRT today, and use the right inference tools to develop AI for any application
on any platform.

[Download Now
][73]

[1]: https://developer.nvidia.com/tensorrt/download
[2]: https://docs.nvidia.com/deeplearning/tensorrt/
[3]: https://github.com/NVIDIA/TensorRT
[4]: https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/
[5]: https://www.nvidia.com/en-us/on-demand/playlist/playList-53110dbc-c11d-4619-b821-987015090afa/
[6]: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
[7]: https://www.nvidia.com/en-us/solutions/ai/inference/balancing-cost-latency-and-performance-eboo
k/
[8]: https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-a
vailable/
[9]: https://developer.nvidia.com/tensorrt-cloud-program
[10]: https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-t
ensorrt-model-optimizer-now-publicly-available/
[11]: https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch
-tensorrt/
[12]: http://hf.co/blog/optimum-nvidia
[13]: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#fit
[14]: https://github.com/NVIDIA/TensorRT/blob/release/10.9/quickstart/IntroNotebooks/2.%20Using%20Py
Torch%20through%20ONNX.ipynb
[15]: https://www.mathworks.com/help/gpucoder/ug/tensorrt-target.html
[16]: https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/
[17]: https://github.com/NVIDIA/TensorRT-LLM/tree/rel
[18]: https://github.com/NVIDIA/TensorRT-LLM/tree/rel
[19]: https://nvidia.github.io/TensorRT-LLM
[20]: https://developer.nvidia.com/nvidia-tensorrt-download
[21]: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
[22]: /tensorrt-rtx
[23]: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html
[24]: https://github.com/NVIDIA/TensorRT-Model-Optimizer
[25]: https://github.com/NVIDIA/TensorRT-Model-Optimizer
[26]: https://nvidia.github.io/TensorRT-Model-Optimizer
[27]: https://github.com/NVIDIA/TensorRT/blob/release/10.9/quickstart/IntroNotebooks/2.%20Using%20Py
Torch%20through%20ONNX.ipynb
[28]: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
[29]: https://pytorch.org/TensorRT/
[30]: https://nvidia.github.io/TensorRT-Incubator/index.html
[31]: https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/examples
[32]: https://github.com/NVIDIA/TensorRT-Incubator/blob/main/tripy/CONTRIBUTING.md
[33]: https://www.nvidia.com/en-us/data-center/products/ai-enterprise/
[34]: https://enterpriseproductregistration.nvidia.com/?LicType=EVAL&ProductFamily=NVAIEnterprise
[35]: https://developer.nvidia.com/blog/tag/inference-performance/
[36]: https://www.nvidia.com/en-us/data-center/mlperf/
[37]: https://www.nvidia.com/en-us/ai-data-science/generative-ai/
[38]: /deep-learning-performance-training-inference/ai-inference
[39]: https://developer.nvidia.com/blog/nvidia-blackwell-leads-on-new-semianalysis-inferencemax-benc
hmarks/
[40]: /tensorrt-getting-started
[41]: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html
[42]: /blog/speeding-up-deep-learning-inference-using-tensorrt-updated/
[43]: /blog/optimizing-and-serving-models-with-nvidia-tensorrt-and-nvidia-triton/
[44]: https://www.youtube.com/watch?v=SlUouzxBldU
[45]: /tensorrt-getting-started
[46]: https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
[47]: /blog/speeding-up-deep-learning-inference-using-tensorrt-updated/
[48]: /blog/optimizing-and-serving-models-with-nvidia-tensorrt-and-nvidia-triton/
[49]: https://www.youtube.com/watch?v=SlUouzxBldU
[50]: https://docs.omniverse.nvidia.com/simready/latest/sim-needs/synth-data-gen.html
[51]: https://docs.omniverse.nvidia.com/extensions/latest/ext_product-configurator.html
[52]: https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-14+V1
[53]: https://build.nvidia.com/nvidia/conditioning-for-precise-visual-generative-ai
[54]: https://www.youtube.com/watch?v=TU5BMU6iYZ0
[55]: /blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
[56]: https://github.com/NVIDIA/Torch-TensorRT/blob/master/notebooks/ssd-object-detection-demo.ipynb
[57]: https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/00-introduction-to-tripy.html
[58]: https://github.com/NVIDIA/TensorRT-Incubator/blob/main/tripy/notebooks/resnet50.ipynb
[59]: https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/examples/nanogpt
[60]: https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/examples/segment-anything-model-v
2
[61]: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/installing.ht
ml
[62]: https://github.com/NVIDIA/TensorRT-RTX/tree/main
[63]: https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-fo
r-rtx/
[64]: https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-fo
r-rtx/
[65]: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/get-started?tabs=csharp
[66]: https://nvidia.github.io/TensorRT-LLM/torch.html#quantization
[67]: https://nvidia.github.io/TensorRT-LLM/torch.html#quantization
[68]: https://nvidia.github.io/TensorRT-LLM/torch.html#quantization
[69]: https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html
[70]: https://nvidia.github.io/TensorRT-LLM/torch/adding_new_model.html
[71]: https://nvidia.github.io/TensorRT-Model-Optimizer/guides/7_speculative_decoding.html
[72]: https://www.nvidia.com/en-us/support/submit-security-vulnerability/
[73]: https://developer.nvidia.com/tensorrt/download
