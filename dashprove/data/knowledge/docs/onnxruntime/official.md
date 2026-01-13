# Accelerated Edge
# Machine Learning

Production-grade AI engine to speed up training and inferencing in your existing technology stack.

In a rush? Get started easily:

`pip install onnxruntime`
`pip install onnxruntime-genai`

[Interested in using other languages? See the many others we support →][1]

## Trusted By

Toggle scrolling

* [[Deezer]][2]
* [[Hypefactors]][3]
* [[Intelligenza Etica]][4]
* [[Algoriddim]][5]
* [[GoodNotes]][6]
* [[Peakspeed]][7]
* [[Redis]][8]
* [[Xilinx]][9]
* [[Teradata]][10]
* [[InFarm]][11]
* [[Writer]][12]
* [[SAS]][13]
* [[Ant Group]][14]
* [[Unreal Engine]][15]
* [[Autodesk]][16]
* [[Intel]][17]
* [[USDA]][18]
* [[Bazaarvoice]][19]
* [[Hugging Face]][20]
* [[Graiphic]][21]
* [[Topaz Labs]][22]
* [[Navitaire]][23]
* [[ATLAS]][24]
* [[Vespa]][25]
* [[Oracle]][26]
* [[Rockchip]][27]
* [[NVIDIA]][28]
* [[Samtec]][29]
* [[Apache OpenNLP]][30]
* [[Camo]][31]
* [[Pieces]][32]
* [[Cephable]][33]
* [[PTW Dosimetry]][34]
* [[AMD]][35]
* [[ClearBlade]][36]
* [[Adobe]][37]

## Generative AI



Integrate the power of Generative AI and Large language Models (LLMs) in your apps and services with
ONNX Runtime. No matter what language you develop in or what platform you need to run on, you can
make use of state-of-the-art models for image synthesis, text generation, and more.

[Learn more about ONNX Runtime & Generative AI →][38]

### Use ONNX Runtime with your favorite language and get started with the tutorials:

[Quickstart][39] [Tutorials][40] [Install ONNX Runtime][41] [Hardware acceleration][42] [Get started
(Docs)][43]

Python

C#

JS

Java

C++

More..
`import onnxruntime as ort
# Load the model and create InferenceSession
model_path = "path/to/your/onnx/model"
session = ort.InferenceSession(model_path)
# "Load and preprocess the input image inputTensor"
...
# Run inference
outputs = session.run(None, {"input": inputTensor})
print(outputs)`
[Python Docs][44]

## Videos

Check out some of our videos to help you get started!

[What is ONNX Runtime (ORT)? ][45]
[Converting Models to ONNX Format ][46]
[Optimize Training and Inference with ONNX Runtime (ACPT/DeepSpeed) ][47]
[ONNX Runtime YouTube channel →][48]

## Cross-Platform



Do you program in Python? C#? C++? Java? JavaScript? Rust? No problem. ONNX Runtime has you covered
with support for many languages. And it runs on Linux, Windows, Mac, iOS, Android, and even in web
browsers.

## Performance



CPU, GPU, NPU - no matter what hardware you run on, ONNX Runtime optimizes for latency, throughput,
memory utilization, and binary size. In addition to excellent out-of-the-box performance for common
usage patterns, additional [model optimization techniques][49] and runtime configurations are
available to further improve performance for specific use cases and models.

## ONNX Runtime Inferencing

ONNX Runtime powers AI in Microsoft products including Windows, Office, Azure Cognitive Services,
and Bing, as well as in thousands of other projects across the world. ONNX Runtime is
cross-platform, supporting cloud, edge, web, and mobile experiences.

[Learn more about ONNX Runtime Inferencing →][50]

### Web Browsers

Run PyTorch and other ML models in the web browser with ONNX Runtime Web.

### Mobile Devices

Infuse your Android and iOS mobile apps with AI using ONNX Runtime Mobile.

## ONNX Runtime Training

ONNX Runtime reduces costs for large model training and enables on-device training.

[Learn more about ONNX Runtime Training →][51]

### Large Model Training

Accelerate training of popular models, including [Hugging Face][52] models like Llama-2-7b and
curated models from the [Azure AI | Machine Learning Studio][53] model catalog.

### On-Device Training

On-device training with ONNX Runtime lets developers take an inference model and train it locally to
deliver a more personalized and privacy-respecting experience for customers.

Please help us improve ONNX Runtime
by participating in our [customer survey][54].

[1]: ./getting-started
[2]: ./testimonials#Deezer
[3]: ./testimonials#Hypefactors
[4]: ./testimonials#Intelligenza%20Etica
[5]: ./testimonials#Algoriddim
[6]: ./testimonials#Goodnotes
[7]: ./testimonials#Peakspeed
[8]: ./testimonials#Redis
[9]: ./testimonials#Xilinx
[10]: ./testimonials#Teradata
[11]: ./testimonials#InFarm
[12]: ./testimonials#Writer
[13]: ./testimonials#SAS
[14]: ./testimonials#Ant%20Group
[15]: ./testimonials#Unreal%20Engine
[16]: ./testimonials#Autodesk
[17]: ./testimonials#Intel
[18]: ./testimonials#United%20States%20Department%20of%20Agriculture,%20Agricultural%20Research%20Se
rvice
[19]: ./testimonials#Bazaarvoice
[20]: ./testimonials#Hugging%20Face
[21]: ./testimonials#Graiphic
[22]: ./testimonials#Topaz%20Labs
[23]: ./testimonials#Navitaire
[24]: ./testimonials#Atlas%20Experiment
[25]: ./testimonials#Vespa.ai
[26]: ./testimonials#Oracle
[27]: ./testimonials#Rockchip
[28]: ./testimonials#NVIDIA
[29]: ./testimonials#Samtec
[30]: ./testimonials#Apache%20OpenNLP
[31]: ./testimonials#Camo
[32]: ./testimonials#Pieces.app
[33]: ./testimonials#Cephable
[34]: ./testimonials#PTW%20Dosimetry
[35]: ./testimonials#AMD
[36]: ./testimonials#ClearBlade
[37]: ./testimonials#Adobe
[38]: ./generative-ai
[39]: ./getting-started
[40]: ./docs/tutorials
[41]: ./docs/install
[42]: ./docs/execution-providers
[43]: ./docs/get-started
[44]: https://onnxruntime.ai/docs/get-started/with-python
[45]: https://www.youtube-nocookie.com/embed/M4o4YRVba4o?si=LHc-2AhKt3TrY60g
[46]: https://www.youtube-nocookie.com/embed/lRBsmnBE9ZA?si=l5i0Q2P7VtSJyGK1
[47]: https://www.youtube-nocookie.com/embed/lC7d_7waHLM?si=U4252VEd1t5ioZUN
[48]: https://www.youtube.com/@ONNXRuntime
[49]: https://onnxruntime.ai/docs/performance/
[50]: ./inference
[51]: ./training
[52]: https://huggingface.co/
[53]: https://ml.azure.com/
[54]: https://ncv.microsoft.com/UySXuzobM9
