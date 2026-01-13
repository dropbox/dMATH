# Neural Network Compression Framework (NNCF)

[Key Features][1] • [Installation][2] • [Documentation][3] • [Usage][4] • [Tutorials and Samples][5]
• [Third-party integration][6] • [Model Zoo][7]

[[GitHub Release]][8] [[Website]][9] [[Apache License Version 2.0]][10] [[PyPI Downloads]][11]

[[Python]][12] [[Backends]][13] [[OS]][14]

Neural Network Compression Framework (NNCF) provides a suite of post-training and training-time
algorithms for optimizing inference of neural networks in [OpenVINO™][15] with a minimal accuracy
drop.

NNCF is designed to work with models from [PyTorch][16], [TorchFX][17], [ONNX][18] and
[OpenVINO™][19].

NNCF provides [samples][20] that demonstrate the usage of compression algorithms for different use
cases and models. See compression results achievable with the NNCF-powered samples on the [NNCF
Model Zoo page][21].

The framework is organized as a Python* package that can be built and used in a standalone mode. The
framework architecture is unified to make it easy to add different compression algorithms for both
PyTorch deep learning frameworks.


## Key Features

### Post-Training Compression Algorithms

───────────────────────────────┬─────────────┬────────────┬─────────────┬─────────────
Compression algorithm          │OpenVINO     │PyTorch     │TorchFX      │ONNX         
───────────────────────────────┼─────────────┼────────────┼─────────────┼─────────────
[Post-Training                 │Supported    │Supported   │Experimental │Supported    
Quantization][22]              │             │            │             │             
───────────────────────────────┼─────────────┼────────────┼─────────────┼─────────────
[Weights Compression][23]      │Supported    │Supported   │Experimental │Supported    
───────────────────────────────┼─────────────┼────────────┼─────────────┼─────────────
[Activation Sparsity][24]      │Not supported│Experimental│Not supported│Not supported
───────────────────────────────┴─────────────┴────────────┴─────────────┴─────────────

### Training-Time Compression Algorithms

──────────────────────────────────────────────────────────────┬─────────
Compression algorithm                                         │PyTorch  
──────────────────────────────────────────────────────────────┼─────────
[Quantization Aware Training][25]                             │Supported
──────────────────────────────────────────────────────────────┼─────────
[Weight-Only Quantization Aware Training with LoRA and        │Supported
NLS][26]                                                      │         
──────────────────────────────────────────────────────────────┼─────────
[Mixed-Precision Quantization][27]                            │Supported
──────────────────────────────────────────────────────────────┴─────────

* Automatic, configurable model graph transformation to obtain the compressed model.
* Common interface for compression methods.
* GPU-accelerated layers for faster compressed model fine-tuning.
* Distributed training support.
* Git patch for prominent third-party repository ([huggingface-transformers][28]) demonstrating the
  process of integrating NNCF into custom training pipelines.
* Exporting PyTorch compressed models to ONNX* checkpoints compressed models to SavedModel or Frozen
  Graph format, ready to use with [OpenVINO™ toolkit][29].


## Documentation

This documentation covers detailed information about NNCF algorithms and functions needed for the
contribution to NNCF.

The latest user documentation for NNCF is available [here][30].

NNCF API documentation can be found [here][31].


## Usage

### Post-Training Quantization

The NNCF PTQ is the simplest way to apply 8-bit quantization. To run the algorithm you only need
your model and a small (~300 samples) calibration dataset.

[OpenVINO][32] is the preferred backend to run PTQ with, while PyTorch and ONNX are also supported.

OpenVINO
import nncf
import openvino as ov
import torch
from torchvision import datasets, transforms

# Instantiate your uncompressed model
model = ov.Core().read_model("/model_path")

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/path", transform=transforms.Compose([transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(model, calibration_dataset)
PyTorch
import nncf
import torch
from torchvision import datasets, models

# Instantiate your uncompressed model
model = models.mobilenet_v2()

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/path", transform=transforms.Compose([transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(val_dataset)

# Step 1: Initialize the transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(model, calibration_dataset)

**NOTE** If the Post-Training Quantization algorithm does not meet quality requirements you can
fine-tune the quantized pytorch model. You can find an example of the Quantization-Aware training
pipeline for a pytorch model [here][33].

TorchFX
import nncf
import torch.fx
from torchvision import datasets, models

# Instantiate your uncompressed model
model = models.mobilenet_v2()

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/path", transform=transforms.Compose([transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(val_dataset)

# Step 1: Initialize the transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)

# Step 3: Export model to TorchFX
input_shape = (1, 3, 224, 224)
fx_model = torch.export.export_for_training(model, args=(ex_input,)).module()
# or
# fx_model = torch.export.export(model, args=(ex_input,)).module()

# Step 4: Run the quantization pipeline
quantized_fx_model = nncf.quantize(fx_model, calibration_dataset)
ONNX
import onnx
import nncf
import torch
from torchvision import datasets

# Instantiate your uncompressed model
onnx_model = onnx.load_model("/model_path")

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/path", transform=transforms.Compose([transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function
input_name = onnx_model.graph.input[0].name
def transform_fn(data_item):
    images, _ = data_item
    return {input_name: images.numpy()}

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(onnx_model, calibration_dataset)

### Training-Time Quantization

Here is an example of Accuracy Aware Quantization pipeline where model weights and compression
parameters may be fine-tuned to achieve a higher accuracy.

PyTorch
import nncf
import nncf.torch
import torch
from torchvision import datasets, models

# Instantiate your uncompressed model
model = models.mobilenet_v2()

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/path", transform=transforms.Compose([transforms.ToTensor()]))
dataset_loader = torch.utils.data.DataLoader(val_dataset)

# Step 1: Initialize the transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(model, calibration_dataset)

# Now use compressed_model as a usual torch.nn.Module
# to fine-tune compression parameters along with the model weights

# Save quantization modules and the quantized model parameters
checkpoint = {
    'state_dict': model.state_dict(),
    'nncf_config': nncf.torch.get_config(model),
    ... # the rest of the user-defined objects to save
}
torch.save(checkpoint, path_to_checkpoint)

# ...

# Load quantization modules and the quantized model parameters
resuming_checkpoint = torch.load(path_to_checkpoint)
nncf_config = resuming_checkpoint['nncf_config']
state_dict = resuming_checkpoint['state_dict']

quantized_model = nncf.torch.load_from_config(model, nncf_config, example_input)
model.load_state_dict(state_dict)
# ... the rest of the usual PyTorch-powered training pipeline


## Demos, Tutorials and Samples

For a quicker start with NNCF-powered compression, try sample notebooks and scripts presented below.

### Jupyter* Notebook Tutorials and Demos

Ready-to-run Jupyter* notebook tutorials and demos are available to explain and display NNCF
compression algorithms for optimizing models for inference with the OpenVINO Toolkit:

─────────────────────────────────────┬─────────────────────────────────┬─────┬──────────────────────
Notebook Tutorial Name               │Compression Algorithm            │Backe│Domain                
                                     │                                 │nd   │                      
─────────────────────────────────────┼─────────────────────────────────┼─────┼──────────────────────
[BERT Quantization][34]              │Post-Training Quantization       │OpenV│NLP                   
[[Colab]][35]                        │                                 │INO  │                      
─────────────────────────────────────┼─────────────────────────────────┼─────┼──────────────────────
[MONAI Segmentation Model            │Post-Training Quantization       │OpenV│Segmentation          
Quantization][36]                    │                                 │INO  │                      
[[Binder]][37]                       │                                 │     │                      
─────────────────────────────────────┼─────────────────────────────────┼─────┼──────────────────────
[PyTorch Model Quantization][38]     │Post-Training Quantization       │PyTor│Image Classification  
                                     │                                 │ch   │                      
─────────────────────────────────────┼─────────────────────────────────┼─────┼──────────────────────
[YOLOv11 Quantization with Accuracy  │Post-Training Quantization with  │OpenV│Speech-to-Text,       
Control][39]                         │Accuracy Control                 │INO  │Object Detection      
─────────────────────────────────────┼─────────────────────────────────┼─────┼──────────────────────
[PyTorch Training-Time               │Training-Time Compression        │PyTor│Image Classification  
Compression][40]                     │                                 │ch   │                      
─────────────────────────────────────┴─────────────────────────────────┴─────┴──────────────────────

A list of notebooks demonstrating OpenVINO conversion and inference together with NNCF compression
for models from various domains:

────────────────────────┬─────────────────────┬──────┬──────────────────────────────────────────────
Demo Model              │Compression Algorithm│Backen│Domain                                        
                        │                     │d     │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[YOLOv8][41]            │Post-Training        │OpenVI│Object Detection,                             
[[Colab]][42]           │Quantization         │NO    │KeyPoint Detection,                           
                        │                     │      │Instance Segmentation                         
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[EfficientSAM][43]      │Post-Training        │OpenVI│Image Segmentation                            
                        │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[Segment Anything       │Post-Training        │OpenVI│Image Segmentation                            
Model][44]              │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[OneFormer][45]         │Post-Training        │OpenVI│Image Segmentation                            
                        │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[CLIP][46]              │Post-Training        │OpenVI│Image-to-Text                                 
                        │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[BLIP][47]              │Post-Training        │OpenVI│Image-to-Text                                 
                        │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[Latent Consistency     │Post-Training        │OpenVI│Text-to-Image                                 
Model][48]              │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[SDXL-turbo][49]        │Post-Training        │OpenVI│Text-to-Image,                                
                        │Quantization         │NO    │Image-to-Image                                
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[Distil-Whisper][50]    │Post-Training        │OpenVI│Speech-to-Text                                
                        │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[Whisper][51]           │Post-Training        │OpenVI│Speech-to-Text                                
[[Colab]][52]           │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[MMS Speech             │Post-Training        │OpenVI│Speech-to-Text                                
Recognition][53]        │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[Grammar Error          │Post-Training        │OpenVI│NLP, Grammar Correction                       
Correction][54]         │Quantization         │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[LLM Instruction        │Weight Compression   │OpenVI│NLP, Instruction Following                    
Following][55]          │                     │NO    │                                              
────────────────────────┼─────────────────────┼──────┼──────────────────────────────────────────────
[LLM Chat Bots][56]     │Weight Compression   │OpenVI│NLP, Chat Bot                                 
                        │                     │NO    │                                              
────────────────────────┴─────────────────────┴──────┴──────────────────────────────────────────────

### Post-Training Quantization and Weight Compression Examples

Compact scripts demonstrating quantization/weight compression and corresponding inference speed
boost:

───────────────────────────────┬────────────────────────────────────────┬───────┬───────────────────
Example Name                   │Compression Algorithm                   │Backend│Domain             
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[OpenVINO MobileNetV2][57]     │Post-Training Quantization              │OpenVIN│Image              
                               │                                        │O      │Classification     
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[OpenVINO YOLOv8][58]          │Post-Training Quantization              │OpenVIN│Object Detection   
                               │                                        │O      │                   
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[OpenVINO YOLOv8 QwAC][59]     │Post-Training Quantization with Accuracy│OpenVIN│Object Detection   
                               │Control                                 │O      │                   
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[OpenVINO Anomaly              │Post-Training Quantization with Accuracy│OpenVIN│Anomaly            
Classification][60]            │Control                                 │O      │Classification     
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[PyTorch MobileNetV2][61]      │Post-Training Quantization              │PyTorch│Image              
                               │                                        │       │Classification     
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[PyTorch SSD][62]              │Post-Training Quantization              │PyTorch│Object Detection   
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[TorchFX Resnet18][63]         │Post-Training Quantization              │TorchFX│Image              
                               │                                        │       │Classification     
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[ONNX MobileNetV2][64]         │Post-Training Quantization              │ONNX   │Image              
                               │                                        │       │Classification     
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[ONNX YOLOv8 QwAC][65]         │Post-Training Quantization with Accuracy│ONNX   │Object Detection   
                               │Control                                 │       │                   
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[ONNX TinyLlama WC][66]        │Weight Compression                      │ONNX   │LLM                
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[TorchFX TinyLlama WC][67]     │Weight Compression                      │TorchFX│LLM                
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[OpenVINO TinyLlama WC][68]    │Weight Compression                      │OpenVIN│LLM                
                               │                                        │O      │                   
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[OpenVINO TinyLlama WC with    │Weight Compression with Hyperparameters │OpenVIN│LLM                
HS][69]                        │Search                                  │O      │                   
───────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────
[ONNX TinyLlama WC with SE][70]│Weight Compression with Scale Estimation│ONNX   │LLM                
───────────────────────────────┴────────────────────────────────────────┴───────┴───────────────────

### Quantization-Aware Training Examples

─────────────────────┬───────────────────────────┬───────┬────────────────────
Example Name         │Compression Algorithm      │Backend│Domain              
─────────────────────┼───────────────────────────┼───────┼────────────────────
[PyTorch             │Quantization-Aware Training│PyTorch│Image Classification
Resnet18][71]        │                           │       │                    
─────────────────────┼───────────────────────────┼───────┼────────────────────
[PyTorch             │Quantization-Aware Training│PyTorch│Anomaly Detection   
Anomalib][72]        │                           │       │                    
─────────────────────┴───────────────────────────┴───────┴────────────────────


## Third-party Repository Integration

NNCF may be easily integrated into training/evaluation pipelines of third-party repositories.

### Used by

* [HuggingFace Optimum Intel][73]
  
  NNCF is used as a compression backend within the renowned `transformers` repository in HuggingFace
  Optimum Intel. For instance, the command below exports the [Llama-3.2-3B-Instruct][74] model to
  OpenVINO format with INT4-quantized weights:
  
  optimum-cli export openvino -m meta-llama/Llama-3.2-3B-Instruct --weight-format int4 ./Llama-3.2-3
  B-Instruct-int4
* [Ultralytics][75]
  
  NNCF is integrated into the Intel OpenVINO export pipeline, enabling quantization for the exported
  models.
* [ExecuTorch][76]
  
  NNCF is used as primary quantization framework for the [ExecuTorch OpenVINO integration][77].
* [torch.compile][78]
  
  NNCF is used as primary quantization framework for the [torch.compile OpenVINO integration][79].
* [OpenVINO Training Extensions][80]
  
  NNCF is integrated into OpenVINO Training Extensions as a model optimization backend. You can
  train, optimize, and export new models based on available model templates as well as run the
  exported models with OpenVINO.


## Installation Guide

For detailed installation instructions, refer to the [Installation][81] guide.

NNCF can be installed as a regular PyPI package via pip:

pip install nncf

NNCF is also available via [conda][82]:

conda install -c conda-forge nncf

System requirements of NNCF correspond to the used backend. System requirements for each backend and
the matrix of corresponding versions can be found in [installation.md][83].

## NNCF Compressed Model Zoo

List of models and compression results for them can be found at our [NNCF Model Zoo page][84].

## Citing

@article{kozlov2020neural,
    title =   {Neural network compression framework for fast model inference},
    author =  {Kozlov, Alexander and Lazarevich, Ivan and Shamporov, Vasily and Lyalyushkin, Nikolay
 and Gorbachev, Yury},
    journal = {arXiv preprint arXiv:2002.08679},
    year =    {2020}
}

## Contributing Guide

Refer to the [CONTRIBUTING.md][85] file for guidelines on contributions to the NNCF repository.

## Useful links

* [Documentation][86]
* [Examples][87]
* [FAQ][88]
* [Notebooks][89]
* [HuggingFace Optimum Intel][90]
* [OpenVINO Model Optimization Guide][91]
* [OpenVINO Hugging Face page][92]
* [OpenVino Performance Benchmarks page][93]

## Telemetry

NNCF as part of the OpenVINO™ toolkit collects anonymous usage data for the purpose of improving
OpenVINO™ tools. You can opt-out at any time by running the following command in the Python
environment where you have NNCF installed:

`opt_in_out --opt_out`

More information available on [OpenVINO telemetry][94].

[1]: #key-features
[2]: #installation-guide
[3]: #documentation
[4]: #usage
[5]: #demos-tutorials-and-samples
[6]: #third-party-repository-integration
[7]: /openvinotoolkit/nncf/blob/develop/docs/ModelZoo.md
[8]: https://github.com/openvinotoolkit/nncf/releases
[9]: https://docs.openvino.ai/nncf
[10]: /openvinotoolkit/nncf/blob/develop/LICENSE
[11]: https://pypi.org/project/nncf/
[12]: https://camo.githubusercontent.com/5f669e7a7b02911ee9e7fd8efe1a7ac12baed13224cca185ebb548497f5
fc0be/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e31302b2d626c7565
[13]: https://camo.githubusercontent.com/3bfd6c36553e90cb3f15e55f9c24fe86d40ab2b2a283feb57e07851d967
8afb3/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6261636b656e64732d6f70656e76696e6f5f
2537435f7079746f7263685f2537435f6f6e6e785f2d6f72616e6765
[14]: https://camo.githubusercontent.com/d619d02993c31c9fb8c5a244bf1184f47f1720d2ebbff5150a514d52914
42be5/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4f532d4c696e75785f2537435f57696e646f
77735f2537435f4d61634f532d626c7565
[15]: https://docs.openvino.ai
[16]: https://pytorch.org/
[17]: https://pytorch.org/docs/stable/fx.html
[18]: https://onnx.ai/
[19]: https://docs.openvino.ai
[20]: #demos-tutorials-and-samples
[21]: /openvinotoolkit/nncf/blob/develop/docs/ModelZoo.md
[22]: /openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/post_training_quantiza
tion/Usage.md
[23]: /openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Us
age.md
[24]: /openvinotoolkit/nncf/blob/develop/src/nncf/experimental/torch/sparsify_activations/Activation
Sparsity.md
[25]: /openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/quantization_aware_tra
ining/Usage.md
[26]: /openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/quantization_aware_tra
ining_lora/Usage.md
[27]: /openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Legac
yQuantization.md#mixed-precision-quantization
[28]: https://github.com/huggingface/transformers
[29]: https://docs.openvino.ai
[30]: https://docs.openvino.ai/nncf
[31]: https://openvinotoolkit.github.io/nncf/autoapi/nncf/
[32]: https://github.com/openvinotoolkit/openvino
[33]: /openvinotoolkit/nncf/blob/develop/examples/quantization_aware_training/torch/resnet18/README.
md
[34]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/language-quantize-
bert
[35]: https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebo
oks/language-quantize-bert/language-quantize-bert.ipynb
[36]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/ct-segmentation-qu
antize
[37]: https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2Fct-seg
mentation-quantize%2Fct-scan-live-inference.ipynb
[38]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-post-train
ing-quantization-nncf
[39]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov11-quantizati
on-with-accuracy-control
[40]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantizati
on-aware-training
[41]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimizatio
n
[42]: https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebo
oks/yolov8-optimization/yolov8-object-detection.ipynb
[43]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/efficient-sam
[44]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/segment-anything
[45]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/oneformer-segmenta
tion
[46]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/clip-zero-shot-ima
ge-classification
[47]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/blip-visual-langua
ge-processing
[48]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/latent-consistency
-models-image-generation
[49]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/sdxl-turbo
[50]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/distil-whisper-asr
[51]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/whisper-subtitles-
generation
[52]: https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/notebo
oks/whisper-subtitles-generation/whisper-convert.ipynb
[53]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/mms-massively-mult
ilingual-speech
[54]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/grammar-correction
[55]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answe
ring
[56]: https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot
[57]: /openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/mobilenet_v2/R
EADME.md
[58]: /openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/yolov8/README.
md
[59]: /openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/yolov8_quantiz
e_with_accuracy_control/README.md
[60]: /openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/anomaly_stfpm_
quantize_with_accuracy_control/README.md
[61]: /openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/torch/mobilenet_v2/READ
ME.md
[62]: /openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/torch/ssd300_vgg16/READ
ME.md
[63]: /openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/torch_fx/resnet18/READM
E.md
[64]: /openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/onnx/mobilenet_v2/READM
E.md
[65]: /openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/onnx/yolov8_quantize_wi
th_accuracy_control/README.md
[66]: /openvinotoolkit/nncf/blob/develop/examples/llm_compression/onnx/tiny_llama/README.md
[67]: /openvinotoolkit/nncf/blob/develop/examples/llm_compression/torch_fx/tiny_llama/README.md
[68]: /openvinotoolkit/nncf/blob/develop/examples/llm_compression/openvino/tiny_llama/README.md
[69]: /openvinotoolkit/nncf/blob/develop/examples/llm_compression/openvino/tiny_llama_find_hyperpara
ms/README.md
[70]: /openvinotoolkit/nncf/blob/develop/examples/llm_compression/onnx/tiny_llama_scale_estimation/R
EADME.md
[71]: /openvinotoolkit/nncf/blob/develop/examples/quantization_aware_training/torch/resnet18/README.
md
[72]: /openvinotoolkit/nncf/blob/develop/examples/quantization_aware_training/torch/anomalib/README.
md
[73]: https://huggingface.co/docs/optimum/intel/optimization_ov
[74]: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
[75]: https://docs.ultralytics.com/integrations/openvino
[76]: https://github.com/pytorch/executorch/blob/main/examples/openvino/README.md
[77]: https://docs.pytorch.org/executorch/main/build-run-openvino.html
[78]: https://docs.pytorch.org/tutorials/prototype/openvino_quantizer.html
[79]: https://docs.openvino.ai/2025/openvino-workflow/torch-compile.html
[80]: https://github.com/openvinotoolkit/training_extensions
[81]: /openvinotoolkit/nncf/blob/develop/docs/Installation.md
[82]: https://anaconda.org/conda-forge/nncf
[83]: /openvinotoolkit/nncf/blob/develop/docs/Installation.md
[84]: /openvinotoolkit/nncf/blob/develop/docs/ModelZoo.md
[85]: /openvinotoolkit/nncf/blob/develop/CONTRIBUTING.md
[86]: /openvinotoolkit/nncf/blob/develop/docs
[87]: /openvinotoolkit/nncf/blob/develop/examples
[88]: /openvinotoolkit/nncf/blob/develop/docs/FAQ.md
[89]: https://github.com/openvinotoolkit/openvino_notebooks#-model-training
[90]: https://huggingface.co/docs/optimum/intel/optimization_ov
[91]: https://docs.openvino.ai/nncf
[92]: https://huggingface.co/OpenVINO#models
[93]: https://docs.openvino.ai/2025/about-openvino/performance-benchmarks.html
[94]: https://docs.openvino.ai/2025/about-openvino/additional-resources/telemetry.html
