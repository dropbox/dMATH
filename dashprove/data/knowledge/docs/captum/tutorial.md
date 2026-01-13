[[Captum]][1]

* [Docs][2]
* [Tutorials][3]
* [API Reference][4]
* [GitHub][5]

## *›*

**

### Captum Tutorials

* [Overview][6]

### Introduction to Captum

* [Getting started with Captum][7]

### Attribution

* [Interpreting text models][8]
* [Intepreting vision with CIFAR][9]
* [Interpreting vision with Pretrained Models][10]
* [Feature ablation on images with ResNet][11]
* [Interpreting multimodal models][12]
* [Interpreting a regression model of California house prices][13]
* [Interpreting semantic segmentation models][14]
* [Using Captum with torch.distributed][15]
* [Interpreting Deep Learning Recommender Models][16]
* [Interpreting vision and text models with LIME][17]
* [Understanding Llama2 with Captum LLM Attribution][18]
* #### Interpreting BERT
  
  * [Interpreting question answering with BERT Part 1][19]
  * [Interpreting question answering with BERT Part 2][20]

### Robustness

* [Applying robustness attacks and metrics to CIFAR model and dataset][21]

### Concept

* [TCAV for image classification for googlenet model][22]
* [TCAV for NLP sentiment analysis model][23]

### Influential Examples

* [Identifying influential examples and mis-labelled examples with TracInCP][24]

### Captum Insight

* [Getting started with Captum Insights][25]
* [Using Captum Insights with multimodal models (VQA)][26]

# Captum Tutorials

The tutorials here will help you understand and use Captum. They assume that you are familiar with
PyTorch and its basic features.

If you are new to Captum, the easiest way to get started is with the [Getting started with
Captum][27] tutorial.

If you are new to PyTorch, the easiest way to get started is with the [What is PyTorch?][28]
tutorial.

#### Getting started with Captum:

In this tutorial we create and train a simple neural network on the Titanic survival dataset. We
then use Integrated Gradients to analyze feature importance. We then deep dive the network to assess
layer and neuron importance using conductance. Finally, we analyze a specific neuron to understand
feature importance for that specific neuron. Find the tutorial [here][29].

### Attribution

#### Interpreting text models:

In this tutorial we use a pre-trained CNN model for sentiment analysis on an IMDB dataset. We use
Captum and Integrated Gradients to interpret model predictions by show which specific words have
highest attribution to the model output. Find the tutorial [here ][30].

#### Interpreting vision with CIFAR:

This tutorial demonstrates how to use Captum for interpreting vision focused models. First we create
and train (or use a pre-trained) a simple CNN model on the CIFAR dataset. We then interpret the
output of an example with a series of overlays using Integrated Gradients and DeepLIFT. Find the
tutorial [here][31].

#### Interpreting vision with Pretrained models:

Like the CIFAR based tutorial above, this tutorial demonstrates how to use Captum for interpreting
vision-focused models. This tutorial begins with a pretrained resnet18 and VGG16 model and
demonstrates how to use Intergrated Gradients along with Noise Tunnel, GradientShap, Occlusion, and
LRP. Find the tutorial [here][32].

#### Feature ablation on images:

This tutorial demonstrates feature ablation in Captum, applied on images as an example. It leverages
segmentation masks to define ablation groups over the input features. We show how this kind of
analysis helps understanding which parts of the input impacts a certain target in the model. Find
the tutorial [here][33].

#### Interpreting multimodal models:

To demonstrate interpreting multimodal models we have chosen to look at an open source Visual
Question Answer (VQA) model. Using Captum and Integrated Gradients we interpret the output of
several test questions and analyze the attribution scores of the text and visual parts of the model.
Find the tutorial [here][34].

#### Understanding Llama2 with Captum LLM Attribution:

This tutorial demonstrates how to easily use the LLM attribution functionality to interpret the
large langague models (LLM) in text generation. It takes Llama2 as the example and shows the
step-by-step improvements from the basic attribution setting to more advanced techniques. Find the
tutorial [here][35].

#### Interpreting question answering with BERT Part 1:

This tutorial demonstrates how to use Captum to interpret a BERT model for question answering. We
use a pre-trained model from Hugging Face fine-tuned on the SQUAD dataset and show how to use hooks
to examine and better understand embeddings, sub-embeddings, BERT, and attention layers. Find the
tutorial [here][36].

#### Interpreting question answering with BERT Part 2:

In the second part of Bert tutorial we analyze attention matrices using attribution algorithms s.a.
Integrated Gradients. This analysis helps us to identify strong interaction pairs between different
tokens for a specific model prediction. We compare our findings with the [vector norms][37] and show
that attribution scores are more meaningful compared to the vector norms. Find the tutorial
[here][38].

#### Interpreting a regression model of California house prices:

To demonstrate interpreting regression models we have chosen to look at the California house prices
dataset. Using Captum and a variety of attribution methods, we evaluate feature importance as well
as internal attribution to understand the network function. Find the tutorial [here][39].

#### Interpreting a semantic segmentation model:

In this tutorial, we demonstrate applying Captum to a semantic segmentation task to understand what
pixels and regions contribute to the labeling of a particular class. We explore applying GradCAM as
well as Feature Ablation to a pretrained Fully-Convolutional Network model with a ResNet-101
backbone. Find the tutorial [here][40].

#### Using Captum with torch.distributed:

This tutorial provides examples of using Captum with the torch.distributed package and DataParallel,
allowing attributions to be computed in a distributed manner across processors, machines or GPUs.
Find the tutorial [here][41].

#### Intepreting DLRM models with Captum:

This tutorial demonstrates how we use Captum for Deep Learning Recommender Models using[dlrm][42]
model published from facebook research and integrated gradients algorithm. It showcases feature
importance differences for sparse and dense features in predicting clicked and non-clicked Ads. It
also analyzes the importance of feature interaction layer and neuron importances in the final fully
connected layer when predicting clicked Ads. Find the tutorial [here][43].

#### Interpreting vision and text models with LIME:

This tutorial demonstrates how to interpret computer vision and text classification models using
Local Interpretable Model-agnostic Explanations (LIME) algorithm. For vision it uses resnet18 model
to explain image classification based on super-pixels extracted by a segmentation mask. For text it
uses a classification model trained on `AG_NEWS` dataset and explains model predictions based on the
word tokens in the input text. Find the tutorial [here][44].

### Robustness

#### Applying robustness attacks and metrics to CIFAR model and dataset:

This tutorial demonstrates how to apply robustness attacks such as FGSM and PGD as well as
robustness metrics such as MinParamPerturbation and AttackComparator to a model trained on CIFAR
dataset. Apart from that it also demonstrates how robustness techniques can be used in conjunction
with attribution algorithms. Find the tutorial [here][45].

### Concept

#### TCAV for image classification for googlenet model:

This tutorial demonstrates how to apply Testing with Concept Activation Vectors (TCAV) algorithm on
image classification problem. It uses googlenet model and imagenet images to showcase the
effectiveness of TCAV algorithm on interpreting zebra predictions through stripes concepts. Find the
tutorial [here][46].

#### TCAV for NLP sentiment analysis model:

This tutorial demonstrates how to apply TCAV algorithm for a NLP task using movie rating dataset and
a CNN-based binary sentiment classification model. It showcases that `positive adjectives` concept
plays a significant role in predicting positive sentiment. Find the tutorial [here][47].

### Influential Examples

#### Identifying influential examples and mis-labelled examples with TracInCP:

This tutorial demonstrates two use cases of the TracInCP method: providing interpretability by
identifying influential training examples for a given prediction, and identifying mis-labelled
examples. These two use cases are demonstrated using the CIFAR dataset and checkpoints obtained from
training a simple CNN model on it (which can also be downloaded to avoid training). Find the
tutorial [here][48].

### Captum Insight

#### Getting Started with Captum Insights:

This tutorial demonstrates how to use Captum Insights for a vision model in a notebook setting. A
simple pretrained torchvision CNN model is loaded and then used on the CIFAR dataset. Captum
Insights is then loaded to visualize the interpretation of specific examples. Find the tutorial
[here][49].

#### Using Captum Insights with multimodal models (VQA):

This tutorial demonstrates how to use Captum Insights for visualizing attributions of a multimodal
model, particularly an open source Visual Question Answer (VQA) model. Find the tutorial [here][50].
Docs[Introduction][51][Getting Started][52][Tutorials][53][API Reference][54]
Legal[Privacy][55][Terms][56]
Social
[captum][57]
[[Facebook Open Source]][58] Copyright © 2025 Facebook Inc.

[1]: /
[2]: /docs/introduction
[3]: /tutorials/
[4]: /api/
[5]: https://github.com/pytorch/captum
[6]: /tutorials/
[7]: /tutorials/Titanic_Basic_Interpret
[8]: /tutorials/IMDB_TorchText_Interpret
[9]: /tutorials/CIFAR_TorchVision_Interpret
[10]: /tutorials/TorchVision_Interpret
[11]: /tutorials/Resnet_TorchVision_Ablation
[12]: /tutorials/Multimodal_VQA_Interpret
[13]: /tutorials/House_Prices_Regression_Interpret
[14]: /tutorials/Segmentation_Interpret
[15]: /tutorials/Distributed_Attribution
[16]: /tutorials/DLRM_Tutorial
[17]: /tutorials/Image_and_Text_Classification_LIME
[18]: /tutorials/Llama2_LLM_Attribution
[19]: /tutorials/Bert_SQUAD_Interpret
[20]: /tutorials/Bert_SQUAD_Interpret2
[21]: /tutorials/CIFAR_Captum_Robustness
[22]: /tutorials/TCAV_Image
[23]: /tutorials/TCAV_NLP
[24]: /tutorials/TracInCP_Tutorial
[25]: /tutorials/CIFAR_TorchVision_Captum_Insights
[26]: /tutorials/Multimodal_VQA_Captum_Insights
[27]: Titanic_Basic_Interpret
[28]: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tens
or-tutorial-py
[29]: Titanic_Basic_Interpret
[30]: IMDB_TorchText_Interpret
[31]: CIFAR_TorchVision_Interpret
[32]: TorchVision_Interpret
[33]: Resnet_TorchVision_Ablation
[34]: Multimodal_VQA_Interpret
[35]: Llama2_LLM_Attribution
[36]: Bert_SQUAD_Interpret
[37]: https://arxiv.org/pdf/2004.10102.pdf
[38]: Bert_SQUAD_Interpret2
[39]: House_Prices_Regression_Interpret
[40]: Segmentation_Interpret
[41]: Distributed_Attribution
[42]: https://github.com/facebookresearch/dlrm
[43]: DLRM_Tutorial
[44]: Image_and_Text_Classification_LIME
[45]: CIFAR_Captum_Robustness
[46]: TCAV_Image
[47]: TCAV_NLP
[48]: TracInCP_Tutorial
[49]: CIFAR_TorchVision_Captum_Insights
[50]: Multimodal_VQA_Captum_Insights
[51]: /docs/introduction
[52]: /docs/getting_started
[53]: /tutorials/
[54]: /api/
[55]: https://opensource.facebook.com/legal/privacy/
[56]: https://opensource.facebook.com/legal/terms/
[57]: https://github.com/pytorch/captum
[58]: https://opensource.facebook.com/
