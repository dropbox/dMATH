[[Captum]][1]

* [Docs][2]
* [Tutorials][3]
* [API Reference][4]
* [GitHub][5]
Captum

## Model Interpretability for PyTorch

[Introduction][6]
[Get Started][7]
[Tutorials][8]

## Check it out in the intro video

## Key Features

## Multi-Modal

Supports interpretability of models across modalities including vision, text, and more.

## Built on PyTorch

Supports most types of PyTorch models and can be used with minimal modification to the original
neural network.

## Extensible

Open source, generic library for interpretability research. Easily implement and benchmark new
algorithms.

## Get Started

1. #### Install Captum:
   
   via conda (recommended):
   `conda install captum -c pytorch
   `
   via pip:
   `pip install captum
   `
2. #### Create and prepare model:
   
   `import numpy as np
   
   import torch
   import torch.nn as nn
   
   from captum.attr import IntegratedGradients
   
   class ToyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.lin1 = nn.Linear(3, 3)
           self.relu = nn.ReLU()
           self.lin2 = nn.Linear(3, 2)
   
           # initialize weights and biases
           self.lin1.weight = nn.Parameter(torch.arange(-4.0, 5.0).view(3, 3))
           self.lin1.bias = nn.Parameter(torch.zeros(1,3))
           self.lin2.weight = nn.Parameter(torch.arange(-3.0, 3.0).view(2, 3))
           self.lin2.bias = nn.Parameter(torch.ones(1,2))
   
       def forward(self, input):
           return self.lin2(self.relu(self.lin1(input)))
   
   
   model = ToyModel()
   model.eval()
   `
3. #### To make computations deterministic, let's fix random seeds:
   
   `torch.manual_seed(123)
   np.random.seed(123)
   `
4. #### Define input and baseline tensors:
   
   `input = torch.rand(2, 3)
   baseline = torch.zeros(2, 3)
   `
5. #### Select algorithm to instantiate and apply (Integrated Gradients in this example):
   
   `ig = IntegratedGradients(model)
   attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
   print('IG Attributions:', attributions)
   print('Convergence Delta:', delta)
   `
6. #### View Output:
   
   `IG Attributions: tensor([[-0.5922, -1.5497, -1.0067],
                            [ 0.0000, -0.2219, -5.1991]])
   Convergence Delta: tensor([2.3842e-07, -4.7684e-07])
   `
Docs[Introduction][9][Getting Started][10][Tutorials][11][API Reference][12]
Legal[Privacy][13][Terms][14]
Social
[captum][15]
[[Facebook Open Source]][16] Copyright © 2025 Facebook Inc.

[1]: /
[2]: /docs/introduction
[3]: /tutorials/
[4]: /api/
[5]: https://github.com/pytorch/captum
[6]: /docs/introduction.html
[7]: #quickstart
[8]: /tutorials/
[9]: /docs/introduction
[10]: /docs/getting_started
[11]: /tutorials/
[12]: /api/
[13]: https://opensource.facebook.com/legal/privacy/
[14]: https://opensource.facebook.com/legal/terms/
[15]: https://github.com/pytorch/captum
[16]: https://opensource.facebook.com/
