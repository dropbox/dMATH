# Quick Start[¶][1]

TorchMetrics is a collection of 100+ PyTorch metrics implementations and an easy-to-use API to
create custom metrics. It offers:

* A standardized interface to increase reproducibility
* Reduces Boilerplate
* Distributed-training compatible
* Rigorously tested
* Automatic accumulation over batches
* Automatic synchronization between multiple devices

You can use TorchMetrics in any PyTorch model, or within [PyTorch Lightning][2] to enjoy additional
features:

* This means that your data will always be placed on the same device as your metrics.
* Native support for logging metrics in Lightning to reduce even more boilerplate.

## Install[¶][3]

You can install TorchMetrics using pip or conda:

# Python Package Index (PyPI)
pip install torchmetrics
# Conda
conda install -c conda-forge torchmetrics

Eventually if there is a missing PyTorch wheel for your OS or Python version you can simply compile
[PyTorch from source][4]:

# Optional if you do not need compile GPU support
export USE_CUDA=0  # just to keep it simple
# you can install the latest state from master
pip install git+https://github.com/pytorch/pytorch.git
# OR set a particular PyTorch release
pip install git+https://github.com/pytorch/pytorch.git@<release-tag>
# and finalize with installing TorchMetrics
pip install torchmetrics

## Using TorchMetrics[¶][5]

### Functional metrics[¶][6]

Similar to [torch.nn][7], most metrics have both a class-based and a functional version. The
functional versions implement the basic operations required for computing each metric. They are
simple python functions that as input take [torch.tensors][8] and return the corresponding metric as
a `torch.tensor`. The code-snippet below shows a simple example for calculating the accuracy using
the functional interface:

import torch
# import our library
import torchmetrics

# simulate a classification problem
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))

acc = torchmetrics.functional.accuracy(preds, target, task="multiclass", num_classes=5)

### Module metrics[¶][9]

Nearly all functional metrics have a corresponding class-based metric that calls it a functional
counterpart underneath. The class-based metrics are characterized by having one or more internal
metrics states (similar to the parameters of the PyTorch module) that allow them to offer additional
functionalities:

* Accumulation of multiple batches
* Automatic synchronization between multiple devices
* Metric arithmetic

The code below shows how to use the class-based interface:

import torch
# import our library
import torchmetrics

# initialize metric
metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)

n_batches = 10
for i in range(n_batches):
    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))
    # metric on current batch
    acc = metric(preds, target)
    print(f"Accuracy on batch {i}: {acc}")

# metric on all batches using custom accumulation
acc = metric.compute()
print(f"Accuracy on all data: {acc}")

# Resetting internal state such that metric ready for new data
metric.reset()

## Implementing your own metric[¶][10]

Implementing your own metric is as easy as subclassing a [`torch.nn.Module`][11]. Simply, subclass
[`Metric`][12] and do the following:

1. Implement `__init__` where you call `self.add_state` for every internal state that is needed for
   the metrics computations
2. Implement `update` method, where all logic that is necessary for updating metric states go
3. Implement `compute` method, where the final metric computations happens

For practical examples and more info about implementing a metric, please see this [page][13].

### Development Environment[¶][14]

TorchMetrics provides a [Devcontainer][15] configuration for [Visual Studio Code][16] to use a
[Docker container][17] as a pre-configured development environment. This avoids struggles setting up
a development environment and makes them reproducible and consistent. Please follow the
[installation instructions][18] and make yourself familiar with the [container tutorials][19] if you
want to use them. In order to use GPUs, you can enable them within the
`.devcontainer/devcontainer.json` file.

[1]: #quick-start
[2]: https://lightning.ai/docs/pytorch/stable/
[3]: #install
[4]: https://github.com/pytorch/pytorch
[5]: #using-torchmetrics
[6]: #functional-metrics
[7]: https://pytorch.org/docs/2.8/nn
[8]: https://pytorch.org/docs/2.8/tensors.html
[9]: #module-metrics
[10]: #implementing-your-own-metric
[11]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
[12]: ../references/metric.html#torchmetrics.Metric
[13]: implement.html#implement
[14]: #development-environment
[15]: https://code.visualstudio.com/docs/remote/containers
[16]: https://code.visualstudio.com/
[17]: https://www.docker.com/
[18]: https://code.visualstudio.com/docs/remote/containers#_installation
[19]: https://code.visualstudio.com/docs/remote/containers-tutorial
