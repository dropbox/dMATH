[RobustBench][1]

* [Leaderboards][2]
* [Paper][3]
* [FAQ][4]
* [Contribute][5]
* [Model Zoo 🚀][6]
[logo]
RobustBench
A standardized benchmark for adversarial robustness

The goal of **RobustBench** is to systematically track the *real* progress in adversarial
robustness. There are already [more than 3'000 papers][7] on this topic, but it is still unclear
which approaches really work and which only lead to [overestimated robustness][8]. We start from
benchmarking common corruptions, \(\ell_\infty\)- and \(\ell_2\)-robustness since these are the most
studied settings in the literature. We use [AutoAttack][9], an ensemble of white-box and black-box
attacks, to standardize the evaluation (for details see [our paper][10]) of the \(\ell_p\)
robustness and CIFAR-10-C for the evaluation of robustness to common corruptions. Additionally, we
open source the [RobustBench library][11] that contains models used for the leaderboard to
facilitate their usage for downstream applications.

To prevent potential overadaptation of new defenses to AutoAttack, we also welcome external
evaluations based on *adaptive attacks*, especially where AutoAttack [flags][12] a potential
overestimation of robustness. For each model, we are interested in the best known robust accuracy
and see AutoAttack and adaptive attacks as complementary.

** News:**

* **May 2022:** We have extended the common corruptions leaderboard on ImageNet with [3D Common
  Corruptions][13] (ImageNet-3DCC). ImageNet-3DCC evaluation is interesting since (1) it includes
  more realistic corruptions and (2) it can be used to assess generalization of the existing models
  which may have overfitted to ImageNet-C. For a quickstart, click [here][14]. See the new
  leaderboard with ImageNet-C and ImageNet-3DCC [here][15] (also mCE metrics can be found
  [here][16]).
* **May 2022:** We fixed the preprocessing issue for ImageNet corruption evaluations: previously we
  used resize to 256x256 and central crop to 224x224 which wasn't necessary since the ImageNet-C
  images are already 224x224. Note that this changed the ranking between the top-1 and top-2
  entries.

Up-to-date leaderboard based
on 120+ models

Unified access to 80+ state-of-the-art
robust models via Model Zoo

Model Zoo

Check out the [available models][17] and our [Colab tutorials][18].
# !pip install git+https://github.com/RobustBench/robustbench@v0.2.1
from robustbench.utils import load_model
# Load a model from the model zoo
model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra',
                   dataset='cifar10',
                   threat_model='Linf')

# Evaluate the Linf robustness of the model using AutoAttack
from robustbench.eval import benchmark
clean_acc, robust_acc = benchmark(model,
                                  dataset='cifar10',
                                  threat_model='Linf')

Analysis

Check out [our paper][19] with a detailed analysis.
[robustness_vs_venues]
Available Leaderboards
[CIFAR-10 (\( \ell_\infty\))][20] [CIFAR-10 (\( \ell_2\))][21] [CIFAR-10 (Corruptions)][22]
[CIFAR-100 (\( \ell_\infty\))][23] [CIFAR-100 (Corruptions)][24] [ImageNet (\( \ell_\infty\))][25]
[ImageNet (Corruptions: IN-C, IN-3DCC)][26]

Leaderboard: CIFAR-10, \( \ell_\infty = 8/255 \), untargeted attack


1
1
1
1
Leaderboard: CIFAR-10, \( \ell_2 = 0.5 \), untargeted attack


1
1
1
1
Leaderboard: CIFAR-10, Common Corruptions, CIFAR-10-C


1
1
1
1
Leaderboard: CIFAR-100, \( \ell_\infty = 8/255 \), untargeted attack


1
1
1
1
Leaderboard: CIFAR-100, Common Corruptions, CIFAR-100-C


1
1
1
1
Leaderboard: ImageNet, \( \ell_\infty = 4/255 \), untargeted attack


1
1
1
1
Leaderboard: ImageNet, Common Corruptions (ImageNet-C, ImageNet-3DCC)

FAQ

➤ How does the RobustBench leaderboard differ from the [AutoAttack leaderboard][27]? 🤔
The [AutoAttack leaderboard][28] was the starting point of RobustBench. Now only the RobustBench
leaderboard is actively maintained.

➤ How does the RobustBench leaderboard differ from [robust-ml.org][29]? 🤔
[robust-ml.org][30] focuses on *adaptive* evaluations, but we provide a **standardized benchmark**.
Adaptive evaluations have been very useful (e.g., see [Tramer et al., 2020][31]), but they are also
very time-consuming and cannot be standardized by definition. Instead, we argue that one can
estimate robustness accurately mostly *without* adaptive attacks but for this one has to introduce
some restrictions on the considered models (see [our paper][32] for more details). However, we do
welcome adaptive evaluations and we are always interested in showing the best known robust accuracy.

➤ How is it related to libraries like [foolbox][33] / [cleverhans][34] / [advertorch][35]? 🤔
These libraries provide implementations of different *attacks*. Besides the standardized benchmark,
**RobustBench** additionally provides a repository of the most robust models. So you can start using
the robust models in one line of code (see the tutorial [here][36]).

➤ Why is Lp-robustness still interesting? 🤔
There are numerous interesting applications of Lp-robustness that span transfer learning ([Salman et
al. (2020)][37], [Utrera et al. (2020)][38]), interpretability ([Tsipras et al. (2018)][39], [Kaur
et al. (2019)][40], [Engstrom et al. (2019)][41]), security ([Tramèr et al. (2018)][42],
[Saadatpanah et al. (2019)][43]), generalization ([Xie et al. (2019)][44], [Zhu et al. (2019)][45],
[Bochkovskiy et al. (2020)][46]), robustness to unseen perturbations ([Xie et al. (2019)][47], [Kang
et al. (2019)][48]), stabilization of GAN training ([Zhong et al. (2020)][49]).

➤ What about verified adversarial robustness? 🤔
We mostly focus on defenses which improve *empirical* robustness, given the lack of clarity
regarding which approaches really improve robustness and which only make some particular attacks
unsuccessful. However, we do not restrict submissions of verifiably robust models (e.g., we have
[Zhang et al. (2019)][50] in our CIFAR-10 Linf leaderboard). For methods targeting verified
robustness, we encourage the readers to check out [Salman et al. (2019)][51] and [Li et al.
(2020)][52].

➤ What if I have a better attack than the one used in this benchmark? 🤔
We will be happy to add a better attack or any adaptive evaluation that would complement our default
standardized attacks!

Citation

Consider citing our whitepaper if you want to reference our leaderboard or if you are using the
models from the Model Zoo:
@article{croce2020robustbench,
    title={RobustBench: a standardized adversarial robustness benchmark},
    author={Croce, Francesco and Andriushchenko, Maksym and Sehwag, Vikash and Debenedetti, Edoardo 
and Flammarion, Nicolas
    and Chiang, Mung and Mittal, Prateek and Matthias Hein},
    journal={arXiv preprint arXiv:2010.09670},
    year={2020}
}

Contribute to RobustBench!

We welcome any contribution in terms of both new robust models and evaluations. Please check
[here][53] for more details.

Feel free to contact us at [adversarial.benchmark@gmail.com][54]

Maintainers

* [Francesco Croce ][55]
* [Maksym Andriushchenko][56]
* [Vikash Sehwag][57]
* [Edoardo Debenedetti][58]
© 2021, RobustBench; [Icons from Icons8][59]

[1]: ./index.html
[2]: #leaderboard
[3]: https://arxiv.org/abs/2010.09670
[4]: #faq
[5]: #contribute
[6]: https://github.com/RobustBench/robustbench
[7]: https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html
[8]: https://arxiv.org/abs/1802.00420
[9]: https://github.com/fra31/auto-attack
[10]: https://arxiv.org/abs/2010.09670
[11]: https://github.com/RobustBench/robustbench
[12]: https://github.com/fra31/auto-attack/blob/master/flags_doc.md
[13]: https://3dcommoncorruptions.epfl.ch
[14]: https://github.com/RobustBench/robustbench#new-evaluating-robustness-of-imagenet-models-agains
t-3d-common-corruptions-imagenet-3dcc
[15]: https://robustbench.github.io/#div_imagenet_corruptions_heading
[16]: https://github.com/RobustBench/robustbench#corruptions-imagenet-c--imagenet-3dcc
[17]: https://github.com/RobustBench/robustbench#model-zoo
[18]: https://github.com/RobustBench/robustbench#notebooks
[19]: https://arxiv.org/abs/2010.09670
[20]: #div_cifar10_Linf_heading
[21]: #div_cifar10_L2_heading
[22]: #div_cifar10_corruptions_heading
[23]: #div_cifar100_Linf_heading
[24]: #div_cifar100_corruptions_heading
[25]: #div_imagenet_Linf_heading
[26]: #div_imagenet_corruptions_heading
[27]: https://github.com/fra31/auto-attack
[28]: https://github.com/fra31/auto-attack
[29]: https://www.robust-ml.org/
[30]: https://www.robust-ml.org/
[31]: https://arxiv.org/abs/2002.08347
[32]: https://arxiv.org/abs/2010.09670
[33]: https://github.com/bethgelab/foolbox
[34]: https://github.com/tensorflow/cleverhans
[35]: https://github.com/BorealisAI/advertorch
[36]: https://github.com/RobustBench/robustbench#model-zoo-quick-tour
[37]: https://arxiv.org/abs/2007.08489
[38]: https://arxiv.org/abs/2007.05869
[39]: https://arxiv.org/abs/1805.12152
[40]: https://arxiv.org/abs/1910.08640
[41]: https://arxiv.org/abs/1906.00945
[42]: https://arxiv.org/abs/1811.03194
[43]: https://arxiv.org/abs/1906.07153
[44]: https://arxiv.org/abs/1911.09665
[45]: https://arxiv.org/abs/1909.11764
[46]: https://arxiv.org/abs/2004.10934
[47]: https://arxiv.org/abs/1911.09665
[48]: https://arxiv.org/abs/1905.01034
[49]: https://arxiv.org/abs/2008.03364
[50]: https://arxiv.org/abs/1906.06316
[51]: https://arxiv.org/abs/1902.08722
[52]: https://arxiv.org/abs/2009.04131
[53]: https://github.com/RobustBench/robustbench#how-to-contribute
[54]: mailto:adversarial.benchmark@gmail.com
[55]: https://twitter.com/fra__31
[56]: https://people.epfl.ch/maksym.andriushchenko
[57]: https://vsehwag.github.io/
[58]: https://edoardo.science
[59]: https://icons8.com/icon/100413/access
