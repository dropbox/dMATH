# Quick Start Guide[#][1]

This TensorRT Quick Start Guide is a starting point for developers who want to try out the TensorRT
SDK; specifically, it demonstrates how to quickly construct an application to run inference on a
TensorRT engine.

## Introduction[#][2]

NVIDIA TensorRT is an SDK for optimizing trained deep-learning models to enable high-performance
inference. TensorRT contains a deep learning inference optimizer and a runtime for execution.

After you have trained your deep learning model in a framework of your choice, TensorRT enables you
to run it with higher throughput and lower latency.

[[Typical Deep Learning Development Cycle Using TensorRT] ][3]

This section covers the basic installation, conversion, and runtime options available in TensorRT
and when they are best applied.

Here is a quick summary of each chapter:

[Installing TensorRT][4] - We provide multiple, simple ways of installing TensorRT.

[The TensorRT Ecosystem][5] - We describe a simple flowchart to show the different types of
conversion and deployment workflows and discuss their pros and cons.

[Example Deployment Using ONNX][6] - This chapter examines the basic steps to convert and deploy
your model. It introduces concepts used in the rest of the guide and walks you through the decisions
you must make to optimize inference execution.

[ONNX Conversion and Deployment][7] - We provide a broad overview of ONNX exports from PyTorch and
pointers to Jupyter notebooks that provide more detail.

[Using the TensorRT Runtime API][8] - This section provides a tutorial on semantic segmentation of
images using the TensorRT C++ and Python API.

For a higher-level application that allows you to deploy your model quickly, refer to the [NVIDIA
Triton Inference Server Quick Start][9].

## Installing TensorRT[#][10]

There are several installation methods for TensorRT. This section covers the most common options
using:

* A container,
* A Debian file, or
* A standalone `pip` wheel file.

For other ways to install TensorRT, refer to the [Installation Guide][11].

For advanced users who are already familiar with TensorRT and want to get their application running
quickly, who are using an NVIDIA CUDA container, or who want to set up automation, follow the
network repo installation instructions (refer to [Using The NVIDIA CUDA Network Repo For Debian
Installation][12]).

### Container Installation[#][13]

This section introduces the customized virtual machine images (VMI) that NVIDIA publishes and
maintains regularly. NVIDIA NGC-certified public cloud platform users can access specific setup
instructions by browsing the [NGC website][14] and identifying an available NGC container and tag to
run on their VMI.

On each of the major cloud providers, NVIDIA publishes customized GPU-optimized VMIs with regular
updates to OS and drivers. These VMIs are optimized for performance on the latest generations of
NVIDIA GPUs. Using these VMIs to deploy NGC-hosted containers, models, and resources on cloud-hosted
virtual machine instances with B200, B300, H100, A100, L40S, or T4 GPUs ensures optimum performance
for deep learning, machine learning, and HPC workloads.

To deploy a TensorRT container on a public cloud, follow the steps associated with your
[NGC-certified public cloud platform][15].

### Debian Installation[#][16]

Refer to the [Debian Installation][17] instructions.

### Python Package Index Installation[#][18]

Refer to the [Python Package Index Installation][19] instructions.

## The TensorRT Ecosystem[#][20]

TensorRT is a large and flexible project. It can handle a variety of conversion and deployment
workflows, and which workflow is best for you will depend on your specific use case and problem
setting.

TensorRT provides several deployment options, but all workflows involve converting your model to an
optimized representation, which TensorRT refers to as an *engine*. Building a TensorRT workflow for
your model involves picking the right deployment option and the right combination of parameters for
engine creation.

### Basic TensorRT Workflow[#][21]

You must follow five basic steps to convert and deploy your model:

1. Export the model
2. Select a precision
3. Convert the model
4. Deploy the model

It is easiest to understand these steps in the context of a complete, end-to-end workflow: In
[Example Deployment Using ONNX][22], we will cover a simple framework-agnostic deployment workflow
to convert and deploy a trained ResNet-50 model to TensorRT using ONNX conversion and TensorRT’s
standalone runtime.

### Conversion and Deployment Options[#][23]

The TensorRT ecosystem breaks down into two parts:

1. You can follow various paths to convert their models to optimized TensorRT engines.
2. The various runtimes users can target with TensorRT when deploying their optimized TensorRT
   engines.
[[Main Options Available for Conversion and Deployment] ][24]

#### Conversion[#][25]

There are four main options for converting a model with TensorRT:

1. Using Torch-TensorRT
2. Automatic ONNX conversion from `.onnx` files
3. Using the GUI-based tool [Nsight Deep Learning Designer][26]
4. Manually constructing a network using the TensorRT API (either in C++ or Python)

The PyTorch integration (Torch-TensorRT) provides model conversion and a high-level runtime API for
converting PyTorch models. It can fall back to PyTorch implementations where TensorRT does not
support a particular operator. For more information about supported operators, refer to [ONNX
Operator Support][27].

A more performant option for automatic model conversion and deployment is to convert using ONNX.
ONNX is a framework-agnostic option that works with models in TensorFlow, PyTorch, and more.
TensorRT supports automatic conversion from ONNX files using the TensorRT API or `trtexec`, which we
will use in this section. ONNX conversion is all-or-nothing, meaning all operations in your model
must be supported by TensorRT (or you must provide custom plug-ins for unsupported operations). ONNX
conversion results in a singular TensorRT engine that allows less overhead than Torch-TensorRT

In addition to `trtexec`, [Nsight Deep Learning Designer][28] can convert ONNX files into TensorRT
engines. The GUI-based tool provides model visualization and editing, inference performance
profiling, and easy conversions to TensorRT engines for ONNX models. Nsight Deep Learning Designer
automatically downloads TensorRT bits (including CUDA) on demand without requiring a separate
installation of TensorRT.

You can manually construct TensorRT engines using the TensorRT network definition API for the most
performance and customizability possible. This involves building an identical network to your target
model in TensorRT operation by operation, using only TensorRT operations. After a TensorRT network
is created, you will export just the weights of your model from the framework and load them into
your TensorRT network. For this approach, more information about constructing the model using
TensorRT’s network definition API can be found here:

* [Creating A Network Definition From Scratch Using The C++ API][29]
* [Creating A Network Definition From Scratch Using The Python API][30]

#### Deployment[#][31]

There are three options for deploying a model with TensorRT:

1. Deploying within PyTorch
2. Using the standalone TensorRT runtime API
3. Using NVIDIA Triton Inference Server

Your choice for deployment will determine the steps required to convert the model.

When using Torch-TensorRT, the most common deployment option is simply to deploy within PyTorch.
Torch-TensorRT conversion results in a PyTorch graph with TensorRT operations inserted into it. You
can run Torch-TensorRT models like any other PyTorch model using Python.

The TensorRT runtime API allows for the lowest overhead and finest-grained control. However,
operators that TensorRT does not natively support must be implemented as plugins (a library of
prewritten plugins is available on [GitHub: TensorRT plugin][32]). The most common path for
deploying with the runtime API is using ONNX export from a framework, which we cover in the
following section.

Last, NVIDIA Triton Inference Server is open-source inference-serving software that enables teams to
deploy trained AI models from any framework (TensorFlow, TensorRT, PyTorch, ONNX Runtime, or a
custom framework), from local storage or Google Cloud Platform or AWS S3 on any GPU- or CPU-based
infrastructure (cloud, data center, or edge). It is a flexible project with several unique features,
such as concurrent model execution of heterogeneous models and multiple copies of the same model
(multiple model copies can reduce latency further), load balancing, and model analysis. It is a good
option if you must serve your models over HTTP - such as in a cloud inferencing solution. You can
find the [NVIDIA Triton Inference Server home page][33] and the [documentation][34].

### Selecting the Correct Workflow[#][35]

Two of the most important factors in selecting how to convert and deploy your model are:

1. Your choice of framework
2. Your preferred TensorRT runtime to target

For more information on the runtime options available, refer to the Jupyter notebook included with
this guide on [Understanding TensorRT Runtimes][36].

## Example Deployment Using ONNX[#][37]

[ONNX][38] is a framework-agnostic model format that can be exported from most major frameworks,
including TensorFlow and PyTorch. TensorRT provides a library for directly converting ONNX into a
TensorRT engine through the [ONNX-TRT parser][39].

This section will go through the five steps to convert a pre-trained ResNet-50 model from the ONNX
model zoo into a TensorRT engine. Visually, this is the process we will follow:

[[Deployment Process Using ONNX] ][40]

After you understand the basic steps of the TensorRT workflow, you can dive into the more in-depth
Jupyter notebooks (refer to the following topics) for using TensorRT using Torch-TensorRT or ONNX.
Using the PyTorch framework, you can follow along in the introductory Jupyter Notebook [Running this
Guide][41], which covers these workflow steps in more detail.

### Export the Model[#][42]

The main automatic path for TensorRT conversion requires different model formats to convert a model
successfully: The ONNX path requires that models be saved in ONNX.

We use ONNX in this example, so we need an ONNX model. We will use ResNet-50, a basic backbone
vision model that can be used for various purposes. We will perform classification using a
pre-trained ResNet-50 ONNX model included with the [ONNX model zoo][43].

Download a pre-trained ResNet-50 model from the ONNX model zoo using wget and untar it.

wget https://download.onnxruntime.ai/onnx/models/resnet50.tar.gz
tar xzf resnet50.tar.gz

This will unpack a pretrained ResNet-50 `.onnx` file to the path `resnet50/model.onnx`.

In [Exporting To ONNX From PyTorch][44], you can see how we export ONNX models that will work with
this same deployment workflow.

### Select a Precision[#][45]

Inference typically requires less numeric precision than training. With some care, lower precision
can give you faster computation and lower memory consumption without sacrificing any meaningful
accuracy. TensorRT supports FP32, FP16, FP8, BF16, and INT8 precisions, along with limited support
for INT4 weights.

FP32 is most frameworks’ default training precision, so we will start by using it for inference
here.

import numpy as np
PRECISION = np.float32

We set the precision that our TensorRT engine should use at runtime, which we will do in the next
section.

### Convert the Model[#][46]

The ONNX conversion path is one of the most universal and performant paths for automatic TensorRT
conversion. It works for TensorFlow, PyTorch, and many other frameworks.

Several tools help you convert models from ONNX to a TensorRT engine. One common approach is to use
`trtexec` — a command-line tool included with TensorRT that can, among other things, convert ONNX
models to TensorRT engines and profile them.

We can run this conversion as follows:

trtexec --onnx=resnet50/model.onnx --saveEngine=resnet_engine_intro.engine –-stronglyType

This will convert our `resnet50/model.onnx` to a TensorRT engine named `resnet_engine_intro.engine`
using [strong typing][47].

Note

* To tell `trtexec` where to find our ONNX model, use the following option:
  
  --onnx=resnet50/model.onnx
* To tell `trtexec` where to save our optimized TensorRT engine, use the following option:
  
  --saveEngine=resnet_engine_intro.engine

For developers who prefer the ease of a GUI-based tool, [Nsight Deep Learning Designer][48] enables
you to easily convert an ONNX model into a TensorRT engine file. Most of the command-line parameters
for `trtexec` are also available on the GUI of Nsight Deep Learning Designer.

[[TensorRT Export] ][49]

### Deploy the Model[#][50]

After successfully creating our TensorRT engine, we must decide how to run it with TensorRT.

There are two types of TensorRT runtimes: a standalone runtime with C++ and Python bindings and a
native integration into PyTorch. This section will use a simplified wrapper
(`ONNXClassifierWrapper`) that calls the standalone runtime. We will generate a batch of randomized
“dummy” data and use our `ONNXClassifierWrapper` to run inference on that batch. For more
information on TensorRT runtimes, refer to the [Understanding TensorRT Runtimes][51] Jupyter
Notebook.

1. Set up the `ONNXClassifierWrapper` (using the precision we determined in [Select a
   Precision][52]).
   
   from onnx_helper import ONNXClassifierWrapper
   trt_model = ONNXClassifierWrapper("resnet_engine.trt", target_dtype = PRECISION)
2. Generate a dummy batch.
   
   input_shape = (1, 3, 224, 224)
   dummy_input_batch = np.zeros(input_shape , dtype = PRECISION)
3. Feed a batch of data into our engine and get our predictions.
   
   predictions = trt_model.predict(dummy_input_batch)

Note that the wrapper loads and initializes the engine when running the first batch, so this batch
will generally take a while. For more information about batching, refer to the [Batching][53]
section.

For more information about TensorRT APIs, refer to the [NVIDIA TensorRT API Documentation][54]. For
more information on the `ONNXClassifierWrapper`, refer to its implementation on [GitHub:
onnx_helper.py][55].

## ONNX Conversion and Deployment[#][56]

The ONNX interchange format provides a way to export models from many frameworks, including PyTorch,
TensorFlow, and TensorFlow 2, for use with the TensorRT runtime. Importing models using ONNX
requires the operators in your model to be supported by ONNX and for you to supply plug-in
implementations of any operators TensorRT does not support. (A library of plugins for TensorRT can
be found on [GitHub: plugin][57]).

### Exporting with ONNX[#][58]

ONNX models can be easily generated from PyTorch models using PyTorch [export][59].

[Using PyTorch with TensorRT through the ONNX][60] notebook shows how to generate ONNX models from a
PyTorch ResNet-50 model, convert those ONNX models to TensorRT engines using `trtexec`, and use the
TensorRT runtime to feed input to the TensorRT engine at inference time.

#### Exporting to ONNX from PyTorch[#][61]

One approach to converting a PyTorch model to TensorRT is to export a PyTorch model to ONNX and then
convert it into a TensorRT engine. For more details, refer to [Using PyTorch with TensorRT through
ONNX][62]. The notebook will walk you through this path, starting from the below export steps:

[[Exporting ONNX from PyTorch] ][63]

1. Import a ResNet-50 model from `torchvision`. This will load a copy of ResNet-50 with pre-trained
   weights.
   
   import torchvision.models as models
   
   resnet50 = models.resnet50(pretrained=True, progress=False).eval()
   )
2. Save the ONNX file from PyTorch.
   
   Note
   
   We need a batch of data to save our ONNX file from PyTorch. We will use a dummy batch.
   
   import torch
   
   BATCH_SIZE = 32
   dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)
3. Save the ONNX file.
   
   import torch.onnx
   torch.onnx.export(resnet50, dummy_input, "resnet50_pytorch.onnx", verbose=False)
   )

### Converting ONNX to a TensorRT Engine[#][64]

There are three main ways of converting ONNX files to TensorRT engines:

1. Using `trtexec`
2. Using the TensorRT API
3. Using the [Nsight Deep Learning Designer][65] GUI

In this section, we will focus on using `trtexec`. To convert one of the preceding ONNX models to a
TensorRT engine using `trtexec`, we can run this conversion as follows:

trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt

This will convert our `resnet50_onnx_model.onnx` to a TensorRT engine named `resnet_engine.trt`.

### Deploying a TensorRT Engine to the Python Runtime API[#][66]

Several runtimes are available to target with TensorRT. The TensorRT API is a great way to run ONNX
models when performance is important. The following section will deploy a more complex ONNX model
using the TensorRT runtime API in C++ and Python.

In the notebook [Using PyTorch through ONNX][67], you can see how to deploy the preceding model in
Jupyter with the Python runtime API. Another simple option is to use the `ONNXClassifierWrapper`, as
demonstrated in [Deploy the Model][68].

## Using the TensorRT Runtime API[#][69]

One of the most performant and customizable options for model conversion and deployment is the
TensorRT API, which has C++ and Python bindings.

TensorRT includes a standalone runtime with C++ and Python bindings. It is generally more performant
and customizable than Torch-TRT integration and runs in PyTorch. The C++ API has lower overhead, but
the Python API works well with Python data loaders and libraries like NumPy and SciPy and is easier
to use for prototyping, debugging, and testing.

The following tutorial illustrates the semantic segmentation of images using the TensorRT C++ and
Python API. For this task, a fully convolutional model with a ResNet-101 backbone is used. The model
accepts images of arbitrary sizes and produces per-pixel predictions.

The tutorial consists of the following steps:

1. **Set-up**: Launch the test container and generate the TensorRT engine from a PyTorch model
   exported to ONNX and converted using `trtexec`.
2. **C++ runtime API**: Run inference using engine and TensorRT’s C++ API.
3. **Python runtime API**: Run inference using the engine and TensorRT’s Python API.

### Setting Up the Test Container and Building the TensorRT Engine[#][70]

1. Download the source code for this quick start tutorial from the [TensorRT Open Source Software
   repository][71].
   
   $ git clone https://github.com/NVIDIA/TensorRT.git
   $ cd TensorRT/quickstart
2. Convert a [pre-trained FCN-ResNet-101][72] model to ONNX.
   
   Here, we use the export script included with the tutorial to generate an ONNX model and save it
   to `fcn-resnet101.onnx`. For details on ONNX conversion, refer to [ONNX Conversion and
   Deployment][73]. The script also generates a [test image][74] of size 1282x1026 and saves it to
   `input.ppm`.
   
   [[Test Image, Size 1282x1026] ][75]
   
   1. Launch the NVIDIA PyTorch container to run the export script.
      
      $ docker run --rm -it --gpus all -p 8888:8888 -v `pwd`:/workspace -w /workspace/SemanticSegmen
      tation nvcr.io/nvidia/pytorch:20.12-py3 bash
   2. Run the export script to convert the pre-trained model to ONNX.
      
      $ python3 export.py
   
   Note
   
   FCN-ResNet-101 has one input of dimension `[batch, 3, height, width]` and one output of dimension
   `[batch, 21, height, weight]` containing unnormalized probabilities corresponding to predictions
   for 21 class labels. When exporting the model to ONNX, we append an argmax layer at the output to
   produce per-pixel class labels of the highest probability.
3. Build a TensorRT engine from ONNX using the [trtexec][76] tool.
   
   `trtexec` can generate a TensorRT engine from an ONNX model that can then be deployed using the
   TensorRT runtime API. It leverages the [TensorRT ONNX parser][77] to load the ONNX model into a
   TensorRT network graph and the [TensorRT Builder API][78] to generate an optimized engine.
   
   Building an engine can be time-consuming and is usually performed offline.
   
   trtexec --onnx=fcn-resnet101.onnx --saveEngine=fcn-resnet101.engine --optShapes=input:1x3x1026x12
   82
   
   Successful execution should generate an engine file and something similar to `Successful` in the
   command output.
   
   `trtexec` can build TensorRT engines using the configuration options described in the [Commonly
   Used Command-line Flags][79].

### Running an Engine in C++[#][80]

Compile and run the C++ segmentation tutorial within the test container.

$ cd quickstart
$ make
$ ./bin/segmentation_tutorial

The following steps show how to use the [Deserializing A Plan][81] for inference.

1. Deserialize the TensorRT engine from a file. The file contents are read into a buffer and
   deserialized in memory.
   
   std::vector<char> engineData(fsize);
   engineFile.read(engineData.data(), fsize);
   
   std::unique_ptr<nvinfer1::IRuntime> mRuntime{nvinfer1::createInferRuntime(sample::gLogger.getTRTL
   ogger())};
   
   std::unique_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), 
   fsize));
2. A TensorRT execution context encapsulates execution state such as persistent device memory for
   holding intermediate activation tensors during inference.
   
   Since the segmentation model was built with dynamic shapes enabled, the shape of the input must
   be specified for inference execution. The network output shape may be queried to determine the
   corresponding dimensions of the output buffer.
   
   char const* input_name = "input";
   assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
   auto input_dims = nvinfer1::Dims4{1, /* channels */ 3, height, width};
   context->setInputShape(input_name, input_dims);
   auto input_size = util::getMemorySize(input_dims, sizeof(float));
   char const* output_name = "output";
   assert(mEngine->getTensorDataType(output_name) == nvinfer1::DataType::kINT64);
   auto output_dims = context->getTensorShape(output_name);
   auto output_size = util::getMemorySize(output_dims, sizeof(int64_t));
3. In preparation for inference, CUDA device memory is allocated for all inputs and outputs, image
   data is processed and copied into input memory, and a list of engine bindings is generated.
   
   For semantic segmentation, input image data is processed by fitting into a range of `[0, 1]` and
   normalized using mean `[0.485, 0.456, 0.406]` and std deviation `[0.229, 0.224, 0.225]`. Refer to
   the input-preprocessing requirements for the `torchvision` models [GitHub: models][82]. This
   operation is abstracted by the utility class `RGBImageReader`.
   
   void* input_mem{nullptr};
   cudaMalloc(&input_mem, input_size);
   void* output_mem{nullptr};
   cudaMalloc(&output_mem, output_size);
   const std::vector<float> mean{0.485f, 0.456f, 0.406f};
   const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
   auto input_image{util::RGBImageReader(input_filename, input_dims, mean, stddev)};
   input_image.read();
   cudaStream_t stream;
   auto input_buffer = input_image.process();
   cudaMemcpyAsync(input_mem, input_buffer.get(), input_size, cudaMemcpyHostToDevice, stream);
4. Inference execution is kicked off using the context’s `executeV2` or `enqueueV3` methods. After
   the execution, we copy the results to a host buffer and release all device memory allocations.
   
   context->setTensorAddress(input_name, input_mem);
   context->setTensorAddress(output_name, output_mem);
   bool status = context->enqueueV3(stream);
   auto output_buffer = std::unique_ptr<int64_t>{new int64_t[output_size]};
   cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream);
   cudaStreamSynchronize(stream);
   
   cudaFree(input_mem);
   cudaFree(output_mem);
5. A pseudo-color plot of per-pixel class predictions is written to `output.ppm` to visualize the
   results. The utility class `ArgmaxImageWriter` abstracts this.
   
     const int num_classes{21};
     const std::vector<int> palette{
             (0x1 << 25) - 1, (0x1 << 15) - 1, (0x1 << 21) - 1};
     auto output_image{util::ArgmaxImageWriter(output_filename, output_dims, palette, num_classes)};
     int64_t* output_ptr = output_buffer.get();
     std::vector<int32_t> output_buffer_casted(output_size);
     for (size_t i = 0; i < output_size; ++i) {
          output_buffer_casted[i] = static_cast<int32_t>(output_ptr[i]);
     }
     output_image.process(output_buffer_casted.get());
     output_image.write();
   
   For the test image, the expected output is as follows:
   
   .. image:: /images/test-image-output.PNG
           :width: 800
           :alt: Test Image, Size 1282x1026

### Running an Engine in Python[#][83]

1. Install the required Python packages.
   
   $ pip install pycuda
2. Launch Jupyter and use the provided token to log in using a *http://<host-ip-address>:8888*
   browser.
   
   $ jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
3. Open the [tutorial-runtime.ipynb][84] notebook and follow its steps.

The TensorRT Python runtime APIs map directly to the C++ API described in [Running An Engine In
C++][85].

[1]: #quick-start-guide
[2]: #introduction
[3]: ../_images/dl-cycle.png
[4]: #installing-tensorrt
[5]: #ecosystem
[6]: #ex-deploy-onnx
[7]: #onnx-export
[8]: #runtime
[9]: https://github.com/triton-inference-server/server/blob/r20.12/docs/quickstart.md
[10]: #installing-tensorrt
[11]: ../installing-tensorrt/overview.html#install-guide
[12]: ../installing-tensorrt/installing.html#net-repo-install-debian
[13]: #container-installation
[14]: http://ngc.nvidia.com/
[15]: https://docs.nvidia.com/ngc/ngc-deploy-public-cloud/index.html
[16]: #debian-installation
[17]: ../installing-tensorrt/installing.html#installing-debian
[18]: #python-package-index-installation
[19]: ../installing-tensorrt/installing.html#installing-pip
[20]: #the-tensorrt-ecosystem
[21]: #basic-tensorrt-workflow
[22]: #ex-deploy-onnx
[23]: #conversion-and-deployment-options
[24]: ../_images/conversion-opt.png
[25]: #conversion
[26]: https://developer.nvidia.com/nsight-dl-designer
[27]: https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md
[28]: https://developer.nvidia.com/nsight-dl-designer
[29]: ../inference-library/c-api-docs.html#network-definition-scratch-c-api
[30]: ../inference-library/python-api-docs.html#network-definition-scratch-python-api
[31]: #deployment
[32]: https://github.com/NVIDIA/TensorRT/tree/main/plugin
[33]: https://developer.nvidia.com/nvidia-triton-inference-server
[34]: https://github.com/triton-inference-server/server/blob/r22.01/README.md#documentation
[35]: #selecting-the-correct-workflow
[36]: https://github.com/NVIDIA/TensorRT/tree/main/quickstart/IntroNotebooks/5.%20Understanding%20Te
nsorRT%20Runtimes.ipynb
[37]: #example-deployment-using-onnx
[38]: https://github.com/onnx/onnx/blob/main/docs/IR.md
[39]: https://github.com/onnx/onnx-tensorrt
[40]: ../_images/deploy-process-onnx.png
[41]: https://github.com/NVIDIA/TensorRT/tree/main/quickstart/IntroNotebooks/0.%20Running%20This%20G
uide.ipynb
[42]: #export-the-model
[43]: https://github.com/onnx/models
[44]: #export-from-pytorch
[45]: #select-a-precision
[46]: #convert-the-model
[47]: ../architecture/capabilities.html#strong-vs-weak-typing
[48]: https://developer.nvidia.com/nsight-dl-designer
[49]: ../_images/TensorRT-export.png
[50]: #deploy-the-model
[51]: https://github.com/NVIDIA/TensorRT/tree/main/quickstart/IntroNotebooks/5.%20Understanding%20Te
nsorRT%20Runtimes.ipynb
[52]: #select-precision
[53]: ../performance/best-practices.html#batching
[54]: https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
[55]: https://github.com/NVIDIA/TensorRT/blob/HEAD/quickstart/IntroNotebooks/onnx_helper.py
[56]: #onnx-conversion-and-deployment
[57]: https://github.com/NVIDIA/TensorRT/tree/main/plugin
[58]: #exporting-with-onnx
[59]: https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
[60]: https://github.com/NVIDIA/TensorRT/blob/HEAD/quickstart/IntroNotebooks/2.%20Using%20PyTorch%20
through%20ONNX.ipynb
[61]: #exporting-to-onnx-from-pytorch
[62]: https://github.com/NVIDIA/TensorRT/blob/HEAD/quickstart/IntroNotebooks/2.%20Using%20PyTorch%20
through%20ONNX.ipynb
[63]: ../_images/export-onnx-pytorch.png
[64]: #converting-onnx-to-a-tensorrt-engine
[65]: https://developer.nvidia.com/nsight-dl-designer
[66]: #deploying-a-tensorrt-engine-to-the-python-runtime-api
[67]: https://github.com/NVIDIA/TensorRT/blob/HEAD/quickstart/IntroNotebooks/2.%20Using%20PyTorch%20
through%20ONNX.ipynb
[68]: #deploy-engine
[69]: #using-the-tensorrt-runtime-api
[70]: #setting-up-the-test-container-and-building-the-tensorrt-engine
[71]: http://github.com/NVIDIA/TensorRT
[72]: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
[73]: #onnx-export
[74]: https://pytorch.org/assets/images/deeplab1.png
[75]: ../_images/test-container.PNG
[76]: https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec
[77]: https://github.com/onnx/onnx-tensorrt
[78]: ../inference-library/c-api-docs.html#build-engine-c
[79]: ../reference/command-line-programs.html#trtexec-flags
[80]: #running-an-engine-in-c
[81]: ../inference-library/c-api-docs.html#deserialize-plan-c
[82]: https://github.com/pytorch/vision/blob/main/docs/source/models.rst
[83]: #running-an-engine-in-python
[84]: https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.
ipynb
[85]: #run-engine-c
