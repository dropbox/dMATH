# OpenVINO 2025.4[#][1]

* OpenVINO GenAI
  
  Simplify GenAI model deployment!
  
  [Check out our guide][2]
* OpenVINO models on Hugging Face!
  
  Get pre-optimized OpenVINO models, no need to convert!
  
  [Visit Hugging Face][3]
* OpenVINO Model Hub
  
  See performance benchmarks for top AI models!
  
  [Explore now][4]
* OpenVINO via PyTorch 2.0 torch.compile()
  
  Use OpenVINO directly in PyTorch-native applications!
  
  [Learn more][5]


**OpenVINO is an open-source toolkit** for deploying performant AI solutions in the cloud, on-prem,
and on the edge alike. Develop your applications with both generative and conventional AI models,
coming from the most popular model frameworks. Convert, optimize, and run inference utilizing the
full potential of Intel® hardware. There are three main tools in OpenVINO to meet all your
deployment needs:

OpenVINO GenAI

Run and deploy generative AI models

[./openvino-workflow-generative.html][6]
OpenVINO Base Package

Run and deploy conventional AI models

[./openvino-workflow.html][7]
OpenVINO Model Server

Deploy both generative and conventional AI inference on a server

[./model-server/ovms_what_is_openvino_model_server.html][8]

For a quick ramp-up, check out the [OpenVINO Toolkit Cheat Sheet [PDF]][9] and the [OpenVINO GenAI
Quick-start Guide [PDF]][10]

[[openvino diagram] ][11]


## Where to Begin[#][12]

Installation

This guide introduces installation and learning materials for Intel® Distribution of OpenVINO™
toolkit.

[Get Started][13]

Performance Benchmarks

See latest benchmark numbers for OpenVINO and OpenVINO Model Server.

[View data][14]

Framework Compatibility

Load models directly (for TensorFlow, ONNX, PaddlePaddle) or convert to OpenVINO format.

[Load your model][15]

Easy Deployment

Get started in just a few lines of code.

[Run Inference][16]

Serving at scale

Cloud-ready deployments for microservice applications.

[Check out Model Server][17]

Model Compression

Reach for performance with post-training and training-time compression with NNCF.

[Optimize now][18]


## Key Features[#][19]

[See all features][20]

Model Compression

You can either link directly with OpenVINO Runtime to run inference locally or use OpenVINO Model
Server to serve model inference from a separate server or within Kubernetes environment.

Fast & Scalable Deployment

Write an application once, deploy it anywhere, achieving maximum performance from hardware.
Automatic device discovery allows for superior deployment flexibility. OpenVINO Runtime supports
Linux, Windows and MacOS and provides Python, C++ and C API. Use your preferred language and OS.

Lighter Deployment

Designed with minimal external dependencies reduces the application footprint, simplifying
installation and dependency management. Popular package managers enable application dependencies to
be easily installed and upgraded. Custom compilation for your specific model(s) further reduces
final binary size.

Enhanced App Start-Up Time

In applications where fast start-up is required, OpenVINO significantly reduces first-inference
latency by using the CPU for initial inference and then switching to another device once the model
has been compiled and loaded to memory. Compiled models are cached, improving start-up time even
more.

[1]: #openvino-2025-4
[2]: https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html
[3]: https://huggingface.co/OpenVINO
[4]: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/model-hub.html
[5]: https://docs.openvino.ai/2025/openvino-workflow/torch-compile.html
[6]: ./openvino-workflow-generative.html
[7]: ./openvino-workflow.html
[8]: ./model-server/ovms_what_is_openvino_model_server.html
[9]: https://docs.openvino.ai/2025/_static/download/OpenVINO_Quick_Start_Guide.pdf
[10]: https://docs.openvino.ai/2025/_static/download/GenAI_Quick_Start_Guide.pdf
[11]: _images/openvino-overview-diagram.jpg
[12]: #where-to-begin
[13]: get-started/install-openvino.html
[14]: about-openvino/performance-benchmarks.html
[15]: openvino-workflow/model-preparation.html
[16]: openvino-workflow/running-inference.html
[17]: model-server/ovms_what_is_openvino_model_server.html
[18]: openvino-workflow/model-optimization.html
[19]: #key-features
[20]: about-openvino/key-features.html
