# RobustBench: a standardized adversarial robustness benchmark

**Francesco Croce* (University of Tübingen), Maksym Andriushchenko* (EPFL), Vikash Sehwag*
(Princeton University), Edoardo Debenedetti* (EPFL), Nicolas Flammarion (EPFL), Mung Chiang (Purdue
University), Prateek Mittal (Princeton University), Matthias Hein (University of Tübingen)**

**Leaderboard**: [https://robustbench.github.io/][1]

**Paper:** [https://arxiv.org/abs/2010.09670][2]

**❗Note❗: if you experience problems with the automatic downloading of the models from Google
Drive, install the latest version of `RobustBench` via `pip install
git+https://github.com/RobustBench/robustbench.git`.**




## News

* **May 2022**: We have extended the common corruptions leaderboard on ImageNet with [3D Common
  Corruptions][3] (ImageNet-3DCC). ImageNet-3DCC evaluation is interesting since (1) it includes
  more realistic corruptions and (2) it can be used to assess generalization of the existing models
  which may have overfitted to ImageNet-C. For a quickstart, click [here][4]. Note that the entries
  in leaderboard are still sorted according to ImageNet-C performance.
* **May 2022**: We fixed the preprocessing issue for ImageNet corruption evaluations: previously we
  used resize to 256x256 and central crop to 224x224 which wasn't necessary since the ImageNet-C
  images are already 224x224 (see [this issue][5]). Note that this changed the ranking between the
  top-1 and top-2 entries.

## Main idea

The goal of **`RobustBench`** is to systematically track the *real* progress in adversarial
robustness. There are already [more than 3'000 papers][6] on this topic, but it is still often
unclear which approaches really work and which only lead to [overestimated robustness][7]. We start
from benchmarking the Linf, L2, and common corruption robustness since these are the most studied
settings in the literature.

Evaluation of the robustness to Lp perturbations *in general* is not straightforward and requires
adaptive attacks ([Tramer et al., (2020)][8]). Thus, in order to establish a reliable *standardized*
benchmark, we need to impose some restrictions on the defenses we consider. In particular, **we
accept only defenses that are (1) have in general non-zero gradients wrt the inputs, (2) have a
fully deterministic forward pass (i.e. no randomness) that (3) does not have an optimization loop.**
Often, defenses that violate these 3 principles only make gradient-based attacks harder but do not
substantially improve robustness ([Carlini et al., (2019)][9]) except those that can present
concrete provable guarantees (e.g. [Cohen et al., (2019)][10]).

To prevent potential overadaptation of new defenses to AutoAttack, we also welcome external
evaluations based on **adaptive attacks**, especially where AutoAttack [flags][11] a potential
overestimation of robustness. For each model, we are interested in the best known robust accuracy
and see AutoAttack and adaptive attacks as complementary to each other.

**`RobustBench`** consists of two parts:

* a website [https://robustbench.github.io/][12] with the leaderboard based on many recent papers
  (plots below 👇)
* a collection of the most robust models, **Model Zoo**, which are easy to use for any downstream
  application (see the tutorial below after FAQ 👇)

## FAQ

**Q**: How does the RobustBench leaderboard differ from the [AutoAttack leaderboard][13]? 🤔
**A**: The [AutoAttack leaderboard][14] was the starting point of RobustBench. Now only the
[RobustBench leaderboard][15] is actively maintained.

**Q**: How does the RobustBench leaderboard differ from [robust-ml.org][16]? 🤔
**A**: [robust-ml.org][17] focuses on *adaptive* evaluations, but we provide a **standardized
benchmark**. Adaptive evaluations have been very useful (e.g., see [Tramer et al., 2020][18]) but
they are also very time-consuming and not standardized by definition. Instead, we argue that one can
estimate robustness accurately mostly *without* adaptive attacks but for this one has to introduce
some restrictions on the considered models. However, we do welcome adaptive evaluations and we are
always interested in showing the best known robust accuracy.

**Q**: How is it related to libraries like `foolbox` / `cleverhans` / `advertorch`? 🤔
**A**: These libraries provide implementations of different *attacks*. Besides the standardized
benchmark, **`RobustBench`** additionally provides a repository of the most robust models. So you
can start using the robust models in one line of code (see the tutorial below 👇).

**Q**: Why is Lp-robustness still interesting? 🤔
**A**: There are numerous interesting applications of Lp-robustness that span transfer learning
([Salman et al. (2020)][19], [Utrera et al. (2020)][20]), interpretability ([Tsipras et al.
(2018)][21], [Kaur et al. (2019)][22], [Engstrom et al. (2019)][23]), security ([Tramèr et al.
(2018)][24], [Saadatpanah et al. (2019)][25]), generalization ([Xie et al. (2019)][26], [Zhu et al.
(2019)][27], [Bochkovskiy et al. (2020)][28]), robustness to unseen perturbations ([Xie et al.
(2019)][29], [Kang et al. (2019)][30]), stabilization of GAN training ([Zhong et al. (2020)][31]).

**Q**: What about verified adversarial robustness? 🤔
**A**: We mostly focus on defenses which improve empirical robustness, given the lack of clarity
regarding which approaches really improve robustness and which only make some particular attacks
unsuccessful. However, we do not restrict submissions of verifiably robust models (e.g., we have
[Zhang et al. (2019)][32] in our CIFAR-10 Linf leaderboard). For methods targeting verified
robustness, we encourage the readers to check out [Salman et al. (2019)][33] and [Li et al.
(2020)][34].

**Q**: What if I have a better attack than the one used in this benchmark? 🤔
**A**: We will be happy to add a better attack or any adaptive evaluation that would complement our
default standardized attacks.

## Model Zoo: quick tour

The goal of our **Model Zoo** is to simplify the usage of robust models as much as possible. Check
out our Colab notebook here 👉 [RobustBench: quick start][35] for a quick introduction. It is also
summarized below 👇.

First, install the latest version of **`RobustBench`** (recommended):

pip install git+https://github.com/RobustBench/robustbench.git

or the latest *stable* version of **`RobustBench`** (it is possible that automatic downloading of
the models may not work):

pip install git+https://github.com/RobustBench/robustbench.git@v1.0

Now let's try to load CIFAR-10 and some quite robust CIFAR-10 models from [Carmon2019Unlabeled][36]
that achieves 59.53% robust accuracy evaluated with AA under `eps=8/255`:

from robustbench.data import load_cifar10

x_test, y_test = load_cifar10(n_examples=50)

from robustbench.utils import load_model

model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')

Let's try to evaluate the robustness of this model. We can use any favourite library for this. For
example, [FoolBox][37] implements many different attacks. We can start from a simple PGD attack:

!pip install -q foolbox
import foolbox as fb
fmodel = fb.PyTorchModel(model, bounds=(0, 1))

_, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[
8/255])
print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
`>>> Robust accuracy: 58.0%
`

Wonderful! Can we do better with a more accurate attack?

Let's try to evaluate its robustness with a cheap version [AutoAttack][38] from ICML 2020 with 2/4
attacks (only APGD-CE and APGD-DLR):

# autoattack is installed as a dependency of robustbench so there is not need to install it separate
ly
from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', '
apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test)
`>>> initial accuracy: 92.00%
>>> apgd-ce - 1/1 - 19 out of 46 successfully perturbed
>>> robust accuracy after APGD-CE: 54.00% (total time 10.3 s)
>>> apgd-dlr - 1/1 - 1 out of 27 successfully perturbed
>>> robust accuracy after APGD-DLR: 52.00% (total time 17.0 s)
>>> max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
>>> robust accuracy: 52.00%
`

Note that for our standardized evaluation of Linf-robustness we use the *full* version of AutoAttack
which is slower but more accurate (for that just use `adversary = AutoAttack(model, norm='Linf',
eps=8/255)`).

What about other types of perturbations? Is Lp-robustness useful there? We can evaluate the
available models on more general perturbations. For example, let's take images corrupted by fog
perturbations from CIFAR-10-C with the highest level of severity (5). Are different Linf robust
models perform better on them?

from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy

corruptions = ['fog']
x_test, y_test = load_cifar10c(n_examples=1000, corruptions=corruptions, severity=5)

for model_name in ['Standard', 'Engstrom2019Robustness', 'Rice2020Overfitting',
                   'Carmon2019Unlabeled']:
 model = load_model(model_name, dataset='cifar10', threat_model='Linf')
 acc = clean_accuracy(model, x_test, y_test)
 print(f'Model: {model_name}, CIFAR-10-C accuracy: {acc:.1%}')
`>>> Model: Standard, CIFAR-10-C accuracy: 74.4%
>>> Model: Engstrom2019Robustness, CIFAR-10-C accuracy: 38.8%
>>> Model: Rice2020Overfitting, CIFAR-10-C accuracy: 22.0%
>>> Model: Carmon2019Unlabeled, CIFAR-10-C accuracy: 31.1%
`

As we can see, **all** these Linf robust models perform considerably worse than the standard model
on this type of corruptions. This curious phenomenon was first noticed in [Adversarial Examples Are
a Natural Consequence of Test Error in Noise][39] and explained from the frequency perspective in [A
Fourier Perspective on Model Robustness in Computer Vision][40].

However, on average adversarial training *does* help on CIFAR-10-C. One can check this easily by
loading all types of corruptions via `load_cifar10c(n_examples=1000, severity=5)`, and repeating
evaluation on them.

### ***New***: Evaluating robustness of ImageNet models against 3D Common Corruptions
### (ImageNet-3DCC)

3D Common Corruptions (3DCC) is a recent benchmark by [Kar et al. (CVPR 2022)][41] using scene
geometry to generate realistic corruptions. You can evaluate robustness of a standard ResNet-50
against ImageNet-3DCC by following these steps:

1. Download the data from [here][42] using the provided tool. The data will be saved into a folder
   named `ImageNet-3DCC`.
2. Run the sample evaluation script to obtain accuracies and save them in a pickle file:
import torch 
from robustbench.data import load_imagenet3dcc
from robustbench.utils import clean_accuracy, load_model

corruptions_3dcc = ['near_focus', 'far_focus', 'bit_error', 'color_quant', 
                   'flash', 'fog_3d', 'h265_abr', 'h265_crf',
                   'iso_noise', 'low_light', 'xy_motion_blur', 'z_motion_blur'] # 12 corruptions in 
ImageNet-3DCC

device = torch.device("cuda:0")
model = load_model('Standard_R50', dataset='imagenet', threat_model='corruptions').to(device)
for corruption in corruptions_3dcc:
    for s in [1, 2, 3, 4, 5]:  # 5 severity levels
        x_test, y_test = load_imagenet3dcc(n_examples=5000, corruptions=[corruption], severity=s, da
ta_dir=$PATH_IMAGENET_3DCC)
        acc = clean_accuracy(model, x_test.to(device), y_test.to(device), device=device)
        print(f'Model: {model_name}, ImageNet-3DCC corruption: {corruption} severity: {s} accuracy: 
{acc:.1%}')

## Model Zoo

In order to use a model, you just need to know its ID, e.g. **Carmon2019Unlabeled**, and to run:

from robustbench import load_model

model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')

which automatically downloads the model (all models are defined in `model_zoo/models.py`).

Reproducing evaluation of models from the Model Zoo can be done directly from the command line. Here
is an example of an evaluation of `Salman2020Do_R18` model with AutoAttack on ImageNet for
`eps=4/255=0.0156862745`:

python -m robustbench.eval --n_ex=5000 --dataset=imagenet --threat_model=Linf --model_name=Salman202
0Do_R18 --data_dir=/tmldata1/andriush/imagenet --batch_size=128 --eps=0.0156862745

The CIFAR-10, CIFAR-10-C, CIFAR-100, and CIFAR-100-C datasets are downloaded automatically. However,
the ImageNet datasets should be downloaded manually due to their licensing:

* ImageNet: Obtain the download link [here][43] (requires just signing up from an academic email,
  the approval system there is automatic and happens instantly) and then follow the instructions
  [here][44] to extract the validation set in a pytorch-compatible format into folder `val`.
* ImageNet-C: Please visit [here][45] for the instructions.
* ImageNet-3DCC: Download the data from [here][46] using the provided tool. The data will be saved
  into a folder named `ImageNet-3DCC`.

In order to use the models from the Model Zoo, you can find all available model IDs in the tables
below. Note that the full [leaderboard][47] contains a bit more models which we either have not yet
added to the Model Zoo or their authors don't want them to appear in the Model Zoo.

### CIFAR-10

#### Linf, eps=8/255

──┬──────────────┬──────────────────────────────────────────┬─────┬─────┬─────────────────┬─────────
# │Model ID      │Paper                                     │Clean│Robus│Architecture     │Venue    
  │              │                                          │accur│t    │                 │         
  │              │                                          │acy  │accur│                 │         
  │              │                                          │     │acy  │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Bartoldson│*[Adversarial Robustness Limits via       │93.68│73.71│WideResNet-94-16 │ICML 2024
1*│2024Adversaria│Scaling-Law and Human-Alignment           │%    │%    │                 │         
* │l_WRN-94-16**}│Studies][48]*                             │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Amini2024M│*[MeanSparse: Post-Training Robustness    │93.60│73.10│MeanSparse       │arXiv,   
2*│eanSparse_S-WR│Enhancement Through Mean-Centered Feature │%    │%    │WideResNet-94-16 │Jun 2024 
* │N-94-16**}    │Sparsification][49]*                      │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Bartoldson│*[Adversarial Robustness Limits via       │93.11│71.59│WideResNet-82-8  │ICML 2024
3*│2024Adversaria│Scaling-Law and Human-Alignment           │%    │%    │                 │         
* │l_WRN-82-8**} │Studies][50]*                             │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Peng2023Ro│*[Robust Principles: Architectural Design │93.27│71.07│RaWideResNet-70-1│BMVC 2023
4*│bust**}       │Principles for Adversarially Robust       │%    │%    │6                │         
* │              │CNNs][51]*                                │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Wang2023Be│*[Better Diffusion Models Further Improve │93.25│70.69│WideResNet-70-16 │ICML 2023
5*│tter_WRN-70-16│Adversarial Training][52]*                │%    │%    │                 │         
* │**}           │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Bai2024Mix│*[MixedNUTS: Training-Free                │95.19│69.71│ResNet-152 +     │TMLR, Aug
6*│edNUTS**}     │Accuracy-Robustness Balance via           │%    │%    │WideResNet-70-16 │2024     
* │              │Nonlinearly Mixed Classifiers][53]*       │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Amini2024M│*[MeanSparse: Post-Training Robustness    │93.24│68.94│MeanSparse       │arXiv,   
7*│eanSparse_Ra_W│Enhancement Through Mean-Centered Feature │%    │%    │RaWideResNet-70-1│Jun 2024 
* │RN_70_16**}   │Sparsification][54]*                      │     │     │6                │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Bai2023Imp│*[Improving the Accuracy-Robustness       │95.23│68.06│ResNet-152 +     │SIMODS   
8*│roving_edm**} │Trade-off of Classifiers via Adaptive     │%    │%    │WideResNet-70-16 │2024     
* │              │Smoothing][55]*                           │     │     │+ mixing network │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2023Dec│*[Decoupled Kullback-Leibler Divergence   │92.16│67.73│WideResNet-28-10 │NeurIPS  
9*│oupled_WRN-28-│Loss][56]*                                │%    │%    │                 │2024     
* │10**}         │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Wang2023Be│*[Better Diffusion Models Further Improve │92.44│67.31│WideResNet-28-10 │ICML 2023
10│tter_WRN-28-10│Adversarial Training][57]*                │%    │%    │                 │         
**│**}           │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rebuffi202│*[Fixing Data Augmentation to Improve     │92.23│66.56│WideResNet-70-16 │arXiv,   
11│1Fixing_70_16_│Adversarial Robustness][58]*              │%    │%    │                 │Mar 2021 
**│cutmix_extra**│                                          │     │     │                 │         
  │}             │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Gowal2021I│*[Improving Robustness using Generated    │88.74│66.10│WideResNet-70-16 │NeurIPS  
12│mproving_70_16│Data][59]*                                │%    │%    │                 │2021     
**│_ddpm_100m**} │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Gowal2020U│*[Uncovering the Limits of Adversarial    │91.10│65.87│WideResNet-70-16 │arXiv,   
13│ncovering_70_1│Training against Norm-Bounded Adversarial │%    │%    │                 │Oct 2020 
**│6_extra**}    │Examples][60]*                            │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Huang2022R│*[Revisiting Residual Networks for        │91.58│65.79│WideResNet-A4    │arXiv,   
14│evisiting_WRN-│Adversarial Robustness: An Architectural  │%    │%    │                 │Dec. 2022
**│A4**}         │Perspective][61]*                         │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rebuffi202│*[Fixing Data Augmentation to Improve     │88.50│64.58│WideResNet-106-16│arXiv,   
15│1Fixing_106_16│Adversarial Robustness][62]*              │%    │%    │                 │Mar 2021 
**│_cutmix_ddpm**│                                          │     │     │                 │         
  │}             │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rebuffi202│*[Fixing Data Augmentation to Improve     │88.54│64.20│WideResNet-70-16 │arXiv,   
16│1Fixing_70_16_│Adversarial Robustness][63]*              │%    │%    │                 │Mar 2021 
**│cutmix_ddpm**}│                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Kang2021St│*[Stable Neural ODE with Lyapunov-Stable  │93.73│64.20│WideResNet-70-16,│NeurIPS  
17│able**}       │Equilibrium Points for Defending Against  │%    │%    │Neural ODE block │2021     
**│              │Adversarial Attacks][64]*                 │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Xu2023Expl│*[Exploring and Exploiting Decision       │93.69│63.89│WideResNet-28-10 │ICLR 2023
18│oring_WRN-28-1│Boundary Dynamics for Adversarial         │%    │%    │                 │         
**│0**}          │Robustness][65]*                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Gowal2021I│*[Improving Robustness using Generated    │87.50│63.38│WideResNet-28-10 │NeurIPS  
19│mproving_28_10│Data][66]*                                │%    │%    │                 │2021     
**│_ddpm_100m**} │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Pang2022Ro│*[ Robustness and Accuracy Could Be       │89.01│63.35│WideResNet-70-16 │ICML 2022
20│bustness_WRN70│Reconcilable by (Proper) Definition][67]* │%    │%    │                 │         
**│_16**}        │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rade2021He│*[Helper-based Adversarial Training:      │91.47│62.83│WideResNet-34-10 │OpenRevie
21│lper_extra**} │Reducing Excessive Margin to Achieve a    │%    │%    │                 │w, Jun   
**│              │Better Accuracy vs. Robustness            │     │     │                 │2021     
  │              │Trade-off][68]*                           │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Sehwag2021│*[Robust Learning Meets Generative Models:│87.30│62.79│ResNest152       │ICLR 2022
22│Proxy_ResNest1│Can Proxy Distributions Improve           │%    │%    │                 │         
**│52**}         │Adversarial Robustness?][69]*             │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Gowal2020U│*[Uncovering the Limits of Adversarial    │89.48│62.76│WideResNet-28-10 │arXiv,   
23│ncovering_28_1│Training against Norm-Bounded Adversarial │%    │%    │                 │Oct 2020 
**│0_extra**}    │Examples][70]*                            │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Huang2021E│*[Exploring Architectural Ingredients of  │91.23│62.54│WideResNet-34-R  │NeurIPS  
24│xploring_ema**│Adversarially Robust Deep Neural          │%    │%    │                 │2021     
**│}             │Networks][71]*                            │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Huang2021E│*[Exploring Architectural Ingredients of  │90.56│61.56│WideResNet-34-R  │NeurIPS  
25│xploring**}   │Adversarially Robust Deep Neural          │%    │%    │                 │2021     
**│              │Networks][72]*                            │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Dai2021Par│*[Parameterizing Activation Functions for │87.02│61.55│WideResNet-28-10-│arXiv,   
26│ameterizing**}│Adversarial Robustness][73]*              │%    │%    │PSSiLU           │Oct 2021 
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Pang2022Ro│*[ Robustness and Accuracy Could Be       │88.61│61.04│WideResNet-28-10 │ICML 2022
27│bustness_WRN28│Reconcilable by (Proper) Definition][74]* │%    │%    │                 │         
**│_10**}        │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rade2021He│*[Helper-based Adversarial Training:      │88.16│60.97│WideResNet-28-10 │OpenRevie
28│lper_ddpm**}  │Reducing Excessive Margin to Achieve a    │%    │%    │                 │w, Jun   
**│              │Better Accuracy vs. Robustness            │     │     │                 │2021     
  │              │Trade-off][75]*                           │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rebuffi202│*[Fixing Data Augmentation to Improve     │87.33│60.73│WideResNet-28-10 │arXiv,   
29│1Fixing_28_10_│Adversarial Robustness][76]*              │%    │%    │                 │Mar 2021 
**│cutmix_ddpm**}│                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Sridhar202│*[Improving Neural Network Robustness via │86.53│60.41│WideResNet-34-15 │ACC 2022 
30│1Robust_34_15*│Persistency of Excitation][77]*           │%    │%    │                 │         
**│*}            │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Sehwag2021│*[Robust Learning Meets Generative Models:│86.68│60.27│WideResNet-34-10 │ICLR 2022
31│Proxy**}      │Can Proxy Distributions Improve           │%    │%    │                 │         
**│              │Adversarial Robustness?][78]*             │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Wu2020Adve│*[Adversarial Weight Perturbation Helps   │88.25│60.04│WideResNet-28-10 │NeurIPS  
32│rsarial_extra*│Robust Generalization][79]*               │%    │%    │                 │2020     
**│*}            │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Sridhar202│*[Improving Neural Network Robustness via │89.46│59.66│WideResNet-28-10 │ACC 2022 
33│1Robust**}    │Persistency of Excitation][80]*           │%    │%    │                 │         
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Zhang2020G│*[Geometry-aware Instance-reweighted      │89.36│59.64│WideResNet-28-10 │ICLR 2021
34│eometry**}    │Adversarial Training][81]*                │%    │%    │                 │         
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Carmon2019│*[Unlabeled Data Improves Adversarial     │89.69│59.53│WideResNet-28-10 │NeurIPS  
35│Unlabeled**}  │Robustness][82]*                          │%    │%    │                 │2019     
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Gowal2021I│*[Improving Robustness using Generated    │87.35│58.50│PreActResNet-18  │NeurIPS  
36│mproving_R18_d│Data][83]*                                │%    │%    │                 │2021     
**│dpm_100m**}   │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Chen2024Da│*[Data filtering for efficient adversarial│86.10│58.09│WideResNet-34-20 │Pattern  
37│ta_WRN_34_20**│training][84]*                            │%    │%    │                 │Recogniti
**│}             │                                          │     │     │                 │on 2024  
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Addepalli2│*[Scaling Adversarial Training to Large   │85.32│58.04│WideResNet-34-10 │ECCV 2022
38│021Towards_WRN│Perturbation Bounds][85]*                 │%    │%    │                 │         
**│34**}         │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Addepalli2│*[Efficient and Effective Augmentation    │88.71│57.81│WideResNet-34-10 │NeurIPS  
39│022Efficient_W│Strategy for Adversarial Training][86]*   │%    │%    │                 │2022     
**│RN_34_10**}   │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Chen2021LT│*[LTD: Low Temperature Distillation for   │86.03│57.71│WideResNet-34-20 │arXiv,   
40│D_WRN34_20**} │Robust Adversarial Training][87]*         │%    │%    │                 │Nov 2021 
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rade2021He│*[Helper-based Adversarial Training:      │89.02│57.67│PreActResNet-18  │OpenRevie
41│lper_R18_extra│Reducing Excessive Margin to Achieve a    │%    │%    │                 │w, Jun   
**│**}           │Better Accuracy vs. Robustness            │     │     │                 │2021     
  │              │Trade-off][88]*                           │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Jia2022LAS│*[LAS-AT: Adversarial Training with       │85.66│57.61│WideResNet-70-16 │arXiv,   
42│-AT_70_16**}  │Learnable Attack Strategy][89]*           │%    │%    │                 │Mar 2022 
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Debenedett│*[A Light Recipe to Train Robust Vision   │91.73│57.58│XCiT-L12         │arXiv,   
43│i2022Light_XCi│Transformers][90]*                        │%    │%    │                 │Sep 2022 
**│T-L12**}      │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Chen2024Da│*[Data filtering for efficient adversarial│86.54│57.30│WideResNet-34-10 │Pattern  
44│ta_WRN_34_10**│training][91]*                            │%    │%    │                 │Recogniti
**│}             │                                          │     │     │                 │on 2024  
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Debenedett│*[A Light Recipe to Train Robust Vision   │91.30│57.27│XCiT-M12         │arXiv,   
45│i2022Light_XCi│Transformers][92]*                        │%    │%    │                 │Sep 2022 
**│T-M12**}      │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Sehwag2020│*[HYDRA: Pruning Adversarially Robust     │88.98│57.14│WideResNet-28-10 │NeurIPS  
46│Hydra**}      │Neural Networks][93]*                     │%    │%    │                 │2020     
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Gowal2020U│*[Uncovering the Limits of Adversarial    │85.29│57.14│WideResNet-70-16 │arXiv,   
47│ncovering_70_1│Training against Norm-Bounded Adversarial │%    │%    │                 │Oct 2020 
**│6**}          │Examples][94]*                            │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rade2021He│*[Helper-based Adversarial Training:      │86.86│57.09│PreActResNet-18  │OpenRevie
48│lper_R18_ddpm*│Reducing Excessive Margin to Achieve a    │%    │%    │                 │w, Jun   
**│*}            │Better Accuracy vs. Robustness            │     │     │                 │2021     
  │              │Trade-off][95]*                           │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2023Dec│*[Decoupled Kullback-Leibler Divergence   │85.31│57.09│WideResNet-34-10 │NeurIPS  
49│oupled_WRN-34-│Loss][96]*                                │%    │%    │                 │2024     
**│10**}         │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Chen2021LT│*[LTD: Low Temperature Distillation for   │85.21│56.94│WideResNet-34-10 │arXiv,   
50│D_WRN34_10**} │Robust Adversarial Training][97]*         │%    │%    │                 │Nov 2021 
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Gowal2020U│*[Uncovering the Limits of Adversarial    │85.64│56.82│WideResNet-34-20 │arXiv,   
51│ncovering_34_2│Training against Norm-Bounded Adversarial │%    │%    │                 │Oct 2020 
**│0**}          │Examples][98]*                            │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rebuffi202│*[Fixing Data Augmentation to Improve     │83.53│56.66│PreActResNet-18  │arXiv,   
52│1Fixing_R18_dd│Adversarial Robustness][99]*              │%    │%    │                 │Mar 2021 
**│pm**}         │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Wang2020Im│*[Improving Adversarial Robustness        │87.50│56.29│WideResNet-28-10 │ICLR 2020
53│proving**}    │Requires Revisiting Misclassified         │%    │%    │                 │         
**│              │Examples][100]*                           │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Jia2022LAS│*[LAS-AT: Adversarial Training with       │84.98│56.26│WideResNet-34-10 │arXiv,   
54│-AT_34_10**}  │Learnable Attack Strategy][101]*          │%    │%    │                 │Mar 2022 
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Wu2020Adve│*[Adversarial Weight Perturbation Helps   │85.36│56.17│WideResNet-34-10 │NeurIPS  
55│rsarial**}    │Robust Generalization][102]*              │%    │%    │                 │2020     
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Debenedett│*[A Light Recipe to Train Robust Vision   │90.06│56.14│XCiT-S12         │arXiv,   
56│i2022Light_XCi│Transformers][103]*                       │%    │%    │                 │Sep 2022 
**│T-S12**}      │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Sehwag2021│*[Robust Learning Meets Generative Models:│84.59│55.54│ResNet-18        │ICLR 2022
57│Proxy_R18**}  │Can Proxy Distributions Improve           │%    │%    │                 │         
**│              │Adversarial Robustness?][104]*            │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Hendrycks2│*[Using Pre-Training Can Improve Model    │87.11│54.92│WideResNet-28-10 │ICML 2019
58│019Using**}   │Robustness and Uncertainty][105]*         │%    │%    │                 │         
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Pang2020Bo│*[Boosting Adversarial Training with      │85.14│53.74│WideResNet-34-20 │NeurIPS  
59│osting**}     │Hypersphere Embedding][106]*              │%    │%    │                 │2020     
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2020Lea│*[Learnable Boundary Guided Adversarial   │88.70│53.57│WideResNet-34-20 │ICCV 2021
60│rnable_34_20**│Training][107]*                           │%    │%    │                 │         
**│}             │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Zhang2020A│*[Attacks Which Do Not Kill Training Make │84.52│53.51│WideResNet-34-10 │ICML 2020
61│ttacks**}     │Adversarial Learning Stronger][108]*      │%    │%    │                 │         
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rice2020Ov│*[Overfitting in adversarially robust deep│85.34│53.42│WideResNet-34-20 │ICML 2020
62│erfitting**}  │learning][109]*                           │%    │%    │                 │         
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Huang2020S│*[Self-Adaptive Training: beyond Empirical│83.48│53.34│WideResNet-34-10 │NeurIPS  
63│elf**}        │Risk Minimization][110]*                  │%    │%    │                 │2020     
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Zhang2019T│*[Theoretically Principled Trade-off      │84.92│53.08│WideResNet-34-10 │ICML 2019
64│heoretically**│between Robustness and Accuracy][111]*    │%    │%    │                 │         
**│}             │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2020Lea│*[Learnable Boundary Guided Adversarial   │88.22│52.86│WideResNet-34-10 │ICCV 2021
65│rnable_34_10**│Training][112]*                           │%    │%    │                 │         
**│}             │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Addepalli2│*[Efficient and Effective Augmentation    │85.71│52.48│ResNet-18        │NeurIPS  
66│022Efficient_R│Strategy for Adversarial Training][113]*  │%    │%    │                 │2022     
**│N18**}        │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Chen2020Ad│*[Adversarial Robustness: From            │86.04│51.56│ResNet-50        │CVPR 2020
67│versarial**}  │Self-Supervised Pre-Training to           │%    │%    │(3x ensemble)    │         
**│              │Fine-Tuning][114]*                        │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Chen2020Ef│*[Efficient Robust Training via Backward  │85.32│51.12│WideResNet-34-10 │arXiv,   
68│ficient**}    │Smoothing][115]*                          │%    │%    │                 │Oct 2020 
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Addepalli2│*[Scaling Adversarial Training to Large   │80.24│51.06│ResNet-18        │ECCV 2022
69│021Towards_RN1│Perturbation Bounds][116]*                │%    │%    │                 │         
**│8**}          │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Sitawarin2│*[Improving Adversarial Robustness Through│86.84│50.72│WideResNet-34-10 │arXiv,   
70│020Improving**│Progressive Hardening][117]*              │%    │%    │                 │Mar 2020 
**│}             │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Engstrom20│*[Robustness library][118]*               │87.03│49.25│ResNet-50        │GitHub,  
71│19Robustness**│                                          │%    │%    │                 │Oct 2019 
**│}             │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Zhang2019Y│*[You Only Propagate Once: Accelerating   │87.20│44.83│WideResNet-34-10 │NeurIPS  
72│ou**}         │Adversarial Training via Maximal          │%    │%    │                 │2019     
**│              │Principle][119]*                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Andriushch│*[Understanding and Improving Fast        │79.84│43.93│PreActResNet-18  │NeurIPS  
73│enko2020Unders│Adversarial Training][120]*               │%    │%    │                 │2020     
**│tanding**}    │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Wong2020Fa│*[Fast is better than free: Revisiting    │83.34│43.21│PreActResNet-18  │ICLR 2020
74│st**}         │adversarial training][121]*               │%    │%    │                 │         
**│              │                                          │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Ding2020MM│*[MMA Training: Direct Input Space Margin │84.36│41.44│WideResNet-28-4  │ICLR 2020
75│A**}          │Maximization through Adversarial          │%    │%    │                 │         
**│              │Training][122]*                           │     │     │                 │         
──┼──────────────┼──────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Standard**│*[Standardly trained model][123]*         │94.78│0.00%│WideResNet-28-10 │N/A      
76│}             │                                          │%    │     │                 │         
**│              │                                          │     │     │                 │         
──┴──────────────┴──────────────────────────────────────────┴─────┴─────┴─────────────────┴─────────

#### L2, eps=0.5

──┬───────────────┬──────────────────────────────────────────────┬──────┬──────┬───────────┬────────
# │Model ID       │Paper                                         │Clean │Robust│Architectur│Venue   
  │               │                                              │accura│accura│e          │        
  │               │                                              │cy    │cy    │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Wang2023Bet│*[Better Diffusion Models Further Improve     │95.54%│84.97%│WideResNet-│arXiv,  
1*│ter_WRN-70-16**│Adversarial Training][124]*                   │      │      │70-16      │Feb 2023
* │}              │                                              │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Amini2024Me│*[MeanSparse: Post-Training Robustness        │95.51%│84.33%│MeanSparse │arXiv,  
2*│anSparse_S-WRN-│Enhancement Through Mean-Centered Feature     │      │      │WideResNet-│Jun 2024
* │70-16**}       │Sparsification][125]*                         │      │      │70-16      │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Wang2023Bet│*[Better Diffusion Models Further Improve     │95.16%│83.68%│WideResNet-│ICML    
3*│ter_WRN-28-10**│Adversarial Training][126]*                   │      │      │28-10      │2023    
* │}              │                                              │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Rebuffi2021│*[Fixing Data Augmentation to Improve         │95.74%│82.32%│WideResNet-│arXiv,  
4*│Fixing_70_16_cu│Adversarial Robustness][127]*                 │      │      │70-16      │Mar 2021
* │tmix_extra**}  │                                              │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Gowal2020Un│*[Uncovering the Limits of Adversarial        │94.74%│80.53%│WideResNet-│arXiv,  
5*│covering_extra*│Training against Norm-Bounded Adversarial     │      │      │70-16      │Oct 2020
* │*}             │Examples][128]*                               │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Rebuffi2021│*[Fixing Data Augmentation to Improve         │92.41%│80.42%│WideResNet-│arXiv,  
6*│Fixing_70_16_cu│Adversarial Robustness][129]*                 │      │      │70-16      │Mar 2021
* │tmix_ddpm**}   │                                              │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Rebuffi2021│*[Fixing Data Augmentation to Improve         │91.79%│78.80%│WideResNet-│arXiv,  
7*│Fixing_28_10_cu│Adversarial Robustness][130]*                 │      │      │28-10      │Mar 2021
* │tmix_ddpm**}   │                                              │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Augustin202│*[Adversarial Robustness on In- and           │93.96%│78.79%│WideResNet-│ECCV    
8*│0Adversarial_34│Out-Distribution Improves                     │      │      │34-10      │2020    
* │_10_extra**}   │Explainability][131]*                         │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Sehwag2021P│*[Robust Learning Meets Generative Models: Can│90.93%│77.24%│WideResNet-│ICLR    
9*│roxy**}        │Proxy Distributions Improve Adversarial       │      │      │34-10      │2022    
* │               │Robustness?][132]*                            │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Augustin202│*[Adversarial Robustness on In- and           │92.23%│76.25%│WideResNet-│ECCV    
10│0Adversarial_34│Out-Distribution Improves                     │      │      │34-10      │2020    
**│_10**}         │Explainability][133]*                         │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Rade2021Hel│*[Helper-based Adversarial Training: Reducing │90.57%│76.15%│PreActResNe│OpenRevi
11│per_R18_ddpm**}│Excessive Margin to Achieve a Better Accuracy │      │      │t-18       │ew, Jun 
**│               │vs. Robustness Trade-off][134]*               │      │      │           │2021    
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Rebuffi2021│*[Fixing Data Augmentation to Improve         │90.33%│75.86%│PreActResNe│arXiv,  
12│Fixing_R18_cutm│Adversarial Robustness][135]*                 │      │      │t-18       │Mar 2021
**│ix_ddpm**}     │                                              │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Gowal2020Un│*[Uncovering the Limits of Adversarial        │90.90%│74.50%│WideResNet-│arXiv,  
13│covering**}    │Training against Norm-Bounded Adversarial     │      │      │70-16      │Oct 2020
**│               │Examples][136]*                               │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Sehwag2021P│*[Robust Learning Meets Generative Models: Can│89.76%│74.41%│ResNet-18  │ICLR    
14│roxy_R18**}    │Proxy Distributions Improve Adversarial       │      │      │           │2022    
**│               │Robustness?][137]*                            │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Wu2020Adver│*[Adversarial Weight Perturbation Helps Robust│88.51%│73.66%│WideResNet-│NeurIPS 
15│sarial**}      │Generalization][138]*                         │      │      │34-10      │2020    
**│               │                                              │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Augustin202│*[Adversarial Robustness on In- and           │91.08%│72.91%│ResNet-50  │ECCV    
16│0Adversarial**}│Out-Distribution Improves                     │      │      │           │2020    
**│               │Explainability][139]*                         │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Engstrom201│*[Robustness library][140]*                   │90.83%│69.24%│ResNet-50  │GitHub, 
17│9Robustness**} │                                              │      │      │           │Sep 2019
**│               │                                              │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Rice2020Ove│*[Overfitting in adversarially robust deep    │88.67%│67.68%│PreActResNe│ICML    
18│rfitting**}    │learning][141]*                               │      │      │t-18       │2020    
**│               │                                              │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Rony2019Dec│*[Decoupling Direction and Norm for Efficient │89.05%│66.44%│WideResNet-│CVPR    
19│oupling**}     │Gradient-Based L2 Adversarial Attacks and     │      │      │28-10      │2019    
**│               │Defenses][142]*                               │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Ding2020MMA│*[MMA Training: Direct Input Space Margin     │88.02%│66.09%│WideResNet-│ICLR    
20│**}            │Maximization through Adversarial              │      │      │28-4       │2020    
**│               │Training][143]*                               │      │      │           │        
──┼───────────────┼──────────────────────────────────────────────┼──────┼──────┼───────────┼────────
**│^{**Standard**}│*[Standardly trained model][144]*             │94.78%│0.00% │WideResNet-│N/A     
21│               │                                              │      │      │28-10      │        
**│               │                                              │      │      │           │        
──┴───────────────┴──────────────────────────────────────────────┴──────┴──────┴───────────┴────────

#### Common Corruptions

──┬─────────────────────┬──────────────────────────────────────────┬───────┬───────┬────────┬───────
# │Model ID             │Paper                                     │Clean  │Robust │Architec│Venue  
  │                     │                                          │accurac│accurac│ture    │       
  │                     │                                          │y      │y      │        │       
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Diffenderfer2021W│*[A Winning Hand: Compressing Deep        │96.56% │92.78% │WideResN│NeurIPS
1*│inning_LRR_CARD_Deck*│Networks Can Improve Out-Of-Distribution  │       │       │et-18-2 │2021   
* │*}                   │Robustness][145]*                         │       │       │        │       
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Diffenderfer2021W│*[A Winning Hand: Compressing Deep        │96.66% │90.94% │WideResN│NeurIPS
2*│inning_LRR**}        │Networks Can Improve Out-Of-Distribution  │       │       │et-18-2 │2021   
* │                     │Robustness][146]*                         │       │       │        │       
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Diffenderfer2021W│*[A Winning Hand: Compressing Deep        │95.09% │90.15% │WideResN│NeurIPS
3*│inning_Binary_CARD_De│Networks Can Improve Out-Of-Distribution  │       │       │et-18-2 │2021   
* │ck**}                │Robustness][147]*                         │       │       │        │       
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Kireev2021Effecti│*[On the effectiveness of adversarial     │94.75% │89.60% │ResNet-1│arXiv, 
4*│veness_RLATAugMix**} │training against common corruptions][148]*│       │       │8       │Mar    
* │                     │                                          │       │       │        │2021   
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Hendrycks2020AugM│*[AugMix: A Simple Data Processing Method │95.83% │89.09% │ResNeXt2│ICLR   
5*│ix_ResNeXt**}        │to Improve Robustness and                 │       │       │9_32x4d │2020   
* │                     │Uncertainty][149]*                        │       │       │        │       
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Modas2021PRIMERes│*[PRIME: A Few Primitives Can Boost       │93.06% │89.05% │ResNet-1│arXiv, 
6*│Net18**}             │Robustness to Common Corruptions][150]*   │       │       │8       │Dec    
* │                     │                                          │       │       │        │2021   
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Hendrycks2020AugM│*[AugMix: A Simple Data Processing Method │95.08% │88.82% │WideResN│ICLR   
7*│ix_WRN**}            │to Improve Robustness and                 │       │       │et-40-2 │2020   
* │                     │Uncertainty][151]*                        │       │       │        │       
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Kireev2021Effecti│*[On the effectiveness of adversarial     │94.77% │88.53% │PreActRe│arXiv, 
8*│veness_RLATAugMixNoJS│training against common corruptions][152]*│       │       │sNet-18 │Mar    
* │D**}                 │                                          │       │       │        │2021   
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Diffenderfer2021W│*[A Winning Hand: Compressing Deep        │94.87% │88.32% │WideResN│NeurIPS
9*│inning_Binary**}     │Networks Can Improve Out-Of-Distribution  │       │       │et-18-2 │2021   
* │                     │Robustness][153]*                         │       │       │        │       
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Rebuffi2021Fixing│*[Fixing Data Augmentation to Improve     │95.74% │88.23% │WideResN│arXiv, 
10│_70_16_cutmix_extra_L│Adversarial Robustness][154]*             │       │       │et-70-16│Mar    
**│2**}                 │                                          │       │       │        │2021   
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Kireev2021Effecti│*[On the effectiveness of adversarial     │94.97% │86.60% │PreActRe│arXiv, 
11│veness_AugMixNoJSD**}│training against common corruptions][155]*│       │       │sNet-18 │Mar    
**│                     │                                          │       │       │        │2021   
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Kireev2021Effecti│*[On the effectiveness of adversarial     │93.24% │85.04% │PreActRe│arXiv, 
12│veness_Gauss50percent│training against common corruptions][156]*│       │       │sNet-18 │Mar    
**│**}                  │                                          │       │       │        │2021   
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Kireev2021Effecti│*[On the effectiveness of adversarial     │93.10% │84.10% │PreActRe│arXiv, 
13│veness_RLAT**}       │training against common corruptions][157]*│       │       │sNet-18 │Mar    
**│                     │                                          │       │       │        │2021   
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Rebuffi2021Fixing│*[Fixing Data Augmentation to Improve     │92.23% │82.82% │WideResN│arXiv, 
14│_70_16_cutmix_extra_L│Adversarial Robustness][158]*             │       │       │et-70-16│Mar    
**│inf**}               │                                          │       │       │        │2021   
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Addepalli2022Effi│*[Efficient and Effective Augmentation    │88.71% │80.12% │WideResN│CVPRW  
15│cient_WRN_34_10**}   │Strategy for Adversarial Training][159]*  │       │       │et-34-10│2022   
**│                     │                                          │       │       │        │       
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Addepalli2021Towa│*[Towards Achieving Adversarial Robustness│85.32% │76.78% │WideResN│arXiv, 
16│rds_WRN34**}         │Beyond Perceptual Limits][160]*           │       │       │et-34-10│Apr    
**│                     │                                          │       │       │        │2021   
──┼─────────────────────┼──────────────────────────────────────────┼───────┼───────┼────────┼───────
**│^{**Standard**}      │*[Standardly trained model][161]*         │94.78% │73.46% │WideResN│N/A    
17│                     │                                          │       │       │et-28-10│       
**│                     │                                          │       │       │        │       
──┴─────────────────────┴──────────────────────────────────────────┴───────┴───────┴────────┴───────

### CIFAR-100

#### Linf, eps=8/255

──┬───────────────┬─────────────────────────────────────────┬─────┬─────┬─────────────────┬─────────
# │Model ID       │Paper                                    │Clean│Robus│Architecture     │Venue    
  │               │                                         │accur│t    │                 │         
  │               │                                         │acy  │accur│                 │         
  │               │                                         │     │acy  │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Wang2023Bet│*[Better Diffusion Models Further Improve│75.22│42.66│WideResNet-70-16 │ICML 2023
1*│ter_WRN-70-16**│Adversarial Training][162]*              │%    │%    │                 │         
* │}              │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Amini2024Me│*[MeanSparse: Post-Training Robustness   │75.13│42.25│MeanSparse       │arXiv,   
2*│anSparse_S-WRN-│Enhancement Through Mean-Centered Feature│%    │%    │WideResNet-70-16 │Jun 2024 
* │70-16**}       │Sparsification][163]*                    │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Bai2024Mixe│*[MixedNUTS: Training-Free               │83.08│41.80│ResNet-152 +     │TMLR, Aug
3*│dNUTS**}       │Accuracy-Robustness Balance via          │%    │%    │WideResNet-70-16 │2024     
* │               │Nonlinearly Mixed Classifiers][164]*     │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2023Deco│*[Decoupled Kullback-Leibler Divergence  │73.85│39.18│WideResNet-28-10 │NeurIPS  
4*│upled_WRN-28-10│Loss][165]*                              │%    │%    │                 │2024     
* │**}            │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Wang2023Bet│*[Better Diffusion Models Further Improve│72.58│38.77│WideResNet-28-10 │ICML 2023
5*│ter_WRN-28-10**│Adversarial Training][166]*              │%    │%    │                 │         
* │}              │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Bai2023Impr│*[Improving the Accuracy-Robustness      │85.21│38.72│ResNet-152 +     │SIMODS   
6*│oving_edm**}   │Trade-off of Classifiers via Adaptive    │%    │%    │WideResNet-70-16 │2024     
* │               │Smoothing][167]*                         │     │     │+ mixing network │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Gowal2020Un│*[Uncovering the Limits of Adversarial   │69.15│36.88│WideResNet-70-16 │arXiv,   
7*│covering_extra*│Training against Norm-Bounded Adversarial│%    │%    │                 │Oct 2020 
* │*}             │Examples][168]*                          │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Bai2023Impr│*[Improving the Accuracy-Robustness      │80.18│35.15│ResNet-152 +     │SIMODS   
8*│oving_trades**}│Trade-off of Classifiers via Adaptive    │%    │%    │WideResNet-70-16 │2024     
* │               │Smoothing][169]*                         │     │     │+ mixing network │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Debenedetti│*[A Light Recipe to Train Robust Vision  │70.76│35.08│XCiT-L12         │arXiv,   
9*│2022Light_XCiT-│Transformers][170]*                      │%    │%    │                 │Sep 2022 
* │L12**}         │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rebuffi2021│*[Fixing Data Augmentation to Improve    │63.56│34.64│WideResNet-70-16 │arXiv,   
10│Fixing_70_16_cu│Adversarial Robustness][171]*            │%    │%    │                 │Mar 2021 
**│tmix_ddpm**}   │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Debenedetti│*[A Light Recipe to Train Robust Vision  │69.21│34.21│XCiT-M12         │arXiv,   
11│2022Light_XCiT-│Transformers][172]*                      │%    │%    │                 │Sep 2022 
**│M12**}         │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Pang2022Rob│*[ Robustness and Accuracy Could Be      │65.56│33.05│WideResNet-70-16 │ICML 2022
12│ustness_WRN70_1│Reconcilable by (Proper)                 │%    │%    │                 │         
**│6**}           │Definition][173]*                        │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2023Deco│*[Decoupled Kullback-Leibler Divergence  │65.93│32.52│WideResNet-34-10 │NeurIPS  
13│upled_WRN-34-10│Loss][174]*                              │%    │%    │                 │2024     
**│_autoaug**}    │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Debenedetti│*[A Light Recipe to Train Robust Vision  │67.34│32.19│XCiT-S12         │arXiv,   
14│2022Light_XCiT-│Transformers][175]*                      │%    │%    │                 │Sep 2022 
**│S12**}         │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rebuffi2021│*[Fixing Data Augmentation to Improve    │62.41│32.06│WideResNet-28-10 │arXiv,   
15│Fixing_28_10_cu│Adversarial Robustness][176]*            │%    │%    │                 │Mar 2021 
**│tmix_ddpm**}   │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Jia2022LAS-│*[LAS-AT: Adversarial Training with      │67.31│31.91│WideResNet-34-20 │arXiv,   
16│AT_34_20**}    │Learnable Attack Strategy][177]*         │%    │%    │                 │Mar 2022 
**│               │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2023Deco│*[Decoupled Kullback-Leibler Divergence  │65.76│31.91│WideResNet-34-10 │NeurIPS  
17│upled_WRN-34-10│Loss][178]*                              │%    │%    │                 │2024     
**│**}            │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Addepalli20│*[Efficient and Effective Augmentation   │68.75│31.85│WideResNet-34-10 │NeurIPS  
18│22Efficient_WRN│Strategy for Adversarial Training][179]* │%    │%    │                 │2022     
**│_34_10**}      │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2020Lear│*[Learnable Boundary Guided Adversarial  │62.99│31.20│WideResNet-34-10 │ICCV 2021
19│nable_34_10_LBG│Training][180]*                          │%    │%    │                 │         
**│AT9_eps_8_255**│                                         │     │     │                 │         
  │}              │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Sehwag2021P│*[Robust Learning Meets Generative       │65.93│31.15│WideResNet-34-10 │ICLR 2022
20│roxy**}        │Models: Can Proxy Distributions Improve  │%    │%    │                 │         
**│               │Adversarial Robustness?][181]*           │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Chen2024Dat│*[Data filtering for efficient           │64.32│31.13│WideResNet-34-10 │Pattern  
21│a_WRN_34_10**} │adversarial training][182]*              │%    │%    │                 │Recogniti
**│               │                                         │     │     │                 │on 2024  
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Pang2022Rob│*[ Robustness and Accuracy Could Be      │63.66│31.08│WideResNet-28-10 │ICML 2022
22│ustness_WRN28_1│Reconcilable by (Proper)                 │%    │%    │                 │         
**│0**}           │Definition][183]*                        │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Jia2022LAS-│*[LAS-AT: Adversarial Training with      │64.89│30.77│WideResNet-34-10 │arXiv,   
23│AT_34_10**}    │Learnable Attack Strategy][184]*         │%    │%    │                 │Mar 2022 
**│               │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Chen2021LTD│*[LTD: Low Temperature Distillation for  │64.07│30.59│WideResNet-34-10 │arXiv,   
24│_WRN34_10**}   │Robust Adversarial Training][185]*       │%    │%    │                 │Nov 2021 
**│               │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Addepalli20│*[Scaling Adversarial Training to Large  │65.73│30.35│WideResNet-34-10 │ECCV 2022
25│21Towards_WRN34│Perturbation Bounds][186]*               │%    │%    │                 │         
**│**}            │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2020Lear│*[Learnable Boundary Guided Adversarial  │62.55│30.20│WideResNet-34-20 │ICCV 2021
26│nable_34_20_LBG│Training][187]*                          │%    │%    │                 │         
**│AT6**}         │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Gowal2020Un│*[Uncovering the Limits of Adversarial   │60.86│30.03│WideResNet-70-16 │arXiv,   
27│covering**}    │Training against Norm-Bounded Adversarial│%    │%    │                 │Oct 2020 
**│               │Examples][188]*                          │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2020Lear│*[Learnable Boundary Guided Adversarial  │60.64│29.33│WideResNet-34-10 │ICCV 2021
28│nable_34_10_LBG│Training][189]*                          │%    │%    │                 │         
**│AT6**}         │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rade2021Hel│*[Helper-based Adversarial Training:     │61.50│28.88│PreActResNet-18  │OpenRevie
29│per_R18_ddpm**}│Reducing Excessive Margin to Achieve a   │%    │%    │                 │w, Jun   
**│               │Better Accuracy vs. Robustness           │     │     │                 │2021     
  │               │Trade-off][190]*                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Wu2020Adver│*[Adversarial Weight Perturbation Helps  │60.38│28.86│WideResNet-34-10 │NeurIPS  
30│sarial**}      │Robust Generalization][191]*             │%    │%    │                 │2020     
**│               │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rebuffi2021│*[Fixing Data Augmentation to Improve    │56.87│28.50│PreActResNet-18  │arXiv,   
31│Fixing_R18_ddpm│Adversarial Robustness][192]*            │%    │%    │                 │Mar 2021 
**│**}            │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Hendrycks20│*[Using Pre-Training Can Improve Model   │59.23│28.42│WideResNet-28-10 │ICML 2019
32│19Using**}     │Robustness and Uncertainty][193]*        │%    │%    │                 │         
**│               │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Addepalli20│*[Efficient and Effective Augmentation   │65.45│27.67│ResNet-18        │NeurIPS  
33│22Efficient_RN1│Strategy for Adversarial Training][194]* │%    │%    │                 │2022     
**│8**}           │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Cui2020Lear│*[Learnable Boundary Guided Adversarial  │70.25│27.16│WideResNet-34-10 │ICCV 2021
34│nable_34_10_LBG│Training][195]*                          │%    │%    │                 │         
**│AT0**}         │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Addepalli20│*[Scaling Adversarial Training to Large  │62.02│27.14│PreActResNet-18  │ECCV 2022
35│21Towards_PARN1│Perturbation Bounds][196]*               │%    │%    │                 │         
**│8**}           │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Chen2020Eff│*[Efficient Robust Training via Backward │62.15│26.94│WideResNet-34-10 │arXiv,   
36│icient**}      │Smoothing][197]*                         │%    │%    │                 │Oct 2020 
**│               │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Sitawarin20│*[Improving Adversarial Robustness       │62.82│24.57│WideResNet-34-10 │arXiv,   
37│20Improving**} │Through Progressive Hardening][198]*     │%    │%    │                 │Mar 2020 
**│               │                                         │     │     │                 │         
──┼───────────────┼─────────────────────────────────────────┼─────┼─────┼─────────────────┼─────────
**│^{**Rice2020Ove│*[Overfitting in adversarially robust    │53.83│18.95│PreActResNet-18  │ICML 2020
38│rfitting**}    │deep learning][199]*                     │%    │%    │                 │         
**│               │                                         │     │     │                 │         
──┴───────────────┴─────────────────────────────────────────┴─────┴─────┴─────────────────┴─────────

#### Corruptions

──┬────────────────────┬────────────────────────────────────────┬───────┬───────┬────────┬──────────
# │Model ID            │Paper                                   │Clean  │Robust │Architec│Venue     
  │                    │                                        │accurac│accurac│ture    │          
  │                    │                                        │y      │y      │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Diffenderfer2021│*[A Winning Hand: Compressing Deep      │79.93% │71.08% │WideResN│NeurIPS   
1*│Winning_LRR_CARD_Dec│Networks Can Improve Out-Of-Distribution│       │       │et-18-2 │2021      
* │k**}                │Robustness][200]*                       │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Diffenderfer2021│*[A Winning Hand: Compressing Deep      │78.50% │69.09% │WideResN│NeurIPS   
2*│Winning_Binary_CARD_│Networks Can Improve Out-Of-Distribution│       │       │et-18-2 │2021      
* │Deck**}             │Robustness][201]*                       │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Modas2021PRIMERe│*[PRIME: A Few Primitives Can Boost     │77.60% │68.28% │ResNet-1│arXiv, Dec
3*│sNet18**}           │Robustness to Common Corruptions][202]* │       │       │8       │2021      
* │                    │                                        │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Diffenderfer2021│*[A Winning Hand: Compressing Deep      │78.41% │66.45% │WideResN│NeurIPS   
4*│Winning_LRR**}      │Networks Can Improve Out-Of-Distribution│       │       │et-18-2 │2021      
* │                    │Robustness][203]*                       │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Diffenderfer2021│*[A Winning Hand: Compressing Deep      │77.69% │65.26% │WideResN│NeurIPS   
5*│Winning_Binary**}   │Networks Can Improve Out-Of-Distribution│       │       │et-18-2 │2021      
* │                    │Robustness][204]*                       │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Hendrycks2020Aug│*[AugMix: A Simple Data Processing      │78.90% │65.14% │ResNeXt2│ICLR 2020 
6*│Mix_ResNeXt**}      │Method to Improve Robustness and        │       │       │9_32x4d │          
* │                    │Uncertainty][205]*                      │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Hendrycks2020Aug│*[AugMix: A Simple Data Processing      │76.28% │64.11% │WideResN│ICLR 2020 
7*│Mix_WRN**}          │Method to Improve Robustness and        │       │       │et-40-2 │          
* │                    │Uncertainty][206]*                      │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Addepalli2022Eff│*[Efficient and Effective Augmentation  │68.75% │56.95% │WideResN│CVPRW 2022
8*│icient_WRN_34_10**} │Strategy for Adversarial Training][207]*│       │       │et-34-10│          
* │                    │                                        │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Gowal2020Uncover│*[Uncovering the Limits of Adversarial  │69.15% │56.00% │WideResN│arXiv, Oct
9*│ing_extra_Linf**}   │Training against Norm-Bounded           │       │       │et-70-16│2020      
* │                    │Adversarial Examples][208]*             │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Addepalli2021Tow│*[Towards Achieving Adversarial         │65.73% │54.88% │WideResN│OpenReview
10│ards_WRN34**}       │Robustness Beyond Perceptual            │       │       │et-34-10│, Jun 2021
**│                    │Limits][209]*                           │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Addepalli2021Tow│*[Towards Achieving Adversarial         │62.02% │51.77% │PreActRe│OpenReview
11│ards_PARN18**}      │Robustness Beyond Perceptual            │       │       │sNet-18 │, Jun 2021
**│                    │Limits][210]*                           │       │       │        │          
──┼────────────────────┼────────────────────────────────────────┼───────┼───────┼────────┼──────────
**│^{**Gowal2020Uncover│*[Uncovering the Limits of Adversarial  │60.86% │49.46% │WideResN│arXiv, Oct
12│ing_Linf**}         │Training against Norm-Bounded           │       │       │et-70-16│2020      
**│                    │Adversarial Examples][211]*             │       │       │        │          
──┴────────────────────┴────────────────────────────────────────┴───────┴───────┴────────┴──────────

### ImageNet

*Note:* the values (even clean accuracy) might have small fluctuations depending on the version of
the packages e.g. `torchvision`.

#### Linf, eps=4/255

──┬─────────────────┬────────────────────────────────────────────┬──────┬──────┬─────────┬──────────
# │Model ID         │Paper                                       │Clean │Robust│Architect│Venue     
  │                 │                                            │accura│accura│ure      │          
  │                 │                                            │cy    │cy    │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Xu2024MIMIR_S│*[MIMIR: Masked Image Modeling for Mutual   │78.62%│59.68%│Swin-L   │arXiv, Dec
1*│win-L**}         │Information-based Adversarial               │      │      │         │2023      
* │                 │Robustness][212]*                           │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Liu2023Compre│*[A Comprehensive Study on Robustness of    │78.92%│59.56%│Swin-L   │arXiv, Feb
2*│hensive_Swin-L**}│Image Classification Models: Benchmarking   │      │      │         │2023      
* │                 │and Rethinking][213]*                       │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Amini2024Mean│*[MeanSparse: Post-Training Robustness      │78.80%│58.92%│MeanSpars│arXiv, Jun
3*│Sparse_Swin-L**} │Enhancement Through Mean-Centered Feature   │      │      │e Swin-L │2024      
* │                 │Sparsification][214]*                       │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Bai2024MixedN│*[MixedNUTS: Training-Free                  │81.48%│58.50%│ConvNeXtV│TMLR, Aug 
4*│UTS**}           │Accuracy-Robustness Balance via Nonlinearly │      │      │2-L +    │2024      
* │                 │Mixed Classifiers][215]*                    │      │      │Swin-L   │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Liu2023Compre│*[A Comprehensive Study on Robustness of    │78.02%│58.48%│ConvNeXt-│arXiv, Feb
5*│hensive_ConvNeXt-│Image Classification Models: Benchmarking   │      │      │L        │2023      
* │L**}             │and Rethinking][216]*                       │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Amini2024Mean│*[MeanSparse: Post-Training Robustness      │77.92%│58.22%│MeanSpars│arXiv, Jun
6*│Sparse_ConvNeXt-L│Enhancement Through Mean-Centered Feature   │      │      │e        │2024      
* │**}              │Sparsification][217]*                       │      │      │ConvNeXt-│          
  │                 │                                            │      │      │L        │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Singh2023Revi│*[Revisiting Adversarial Training for       │77.00%│57.70%│ConvNeXt-│NeurIPS   
7*│siting_ConvNeXt-L│ImageNet: Architectures, Training and       │      │      │L +      │2023      
* │-ConvStem**}     │Generalization across Threat Models][218]*  │      │      │ConvStem │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Liu2023Compre│*[A Comprehensive Study on Robustness of    │76.16%│56.16%│Swin-B   │arXiv, Feb
8*│hensive_Swin-B**}│Image Classification Models: Benchmarking   │      │      │         │2023      
* │                 │and Rethinking][219]*                       │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Singh2023Revi│*[Revisiting Adversarial Training for       │75.90%│56.14%│ConvNeXt-│NeurIPS   
9*│siting_ConvNeXt-B│ImageNet: Architectures, Training and       │      │      │B +      │2023      
* │-ConvStem**}     │Generalization across Threat Models][220]*  │      │      │ConvStem │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Xu2024MIMIR_S│*[MIMIR: Masked Image Modeling for Mutual   │76.62%│55.90%│Swin-B   │arXiv, Dec
10│win-B**}         │Information-based Adversarial               │      │      │         │2023      
**│                 │Robustness][221]*                           │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Liu2023Compre│*[A Comprehensive Study on Robustness of    │76.02%│55.82%│ConvNeXt-│arXiv, Feb
11│hensive_ConvNeXt-│Image Classification Models: Benchmarking   │      │      │B        │2023      
**│B**}             │and Rethinking][222]*                       │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Singh2023Revi│*[Revisiting Adversarial Training for       │76.30%│54.66%│ViT-B +  │NeurIPS   
12│siting_ViT-B-Conv│ImageNet: Architectures, Training and       │      │      │ConvStem │2023      
**│Stem**}          │Generalization across Threat Models][223]*  │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**RodriguezMuno│*[Characterizing Model Robustness via       │79.36%│53.82%│Swin-L   │arXiv, Sep
13│z2024Characterizi│Natural Input Gradients][224]*              │      │      │         │2024      
**│ng_Swin-L**}     │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Singh2023Revi│*[Revisiting Adversarial Training for       │74.10%│52.42%│ConvNeXt-│NeurIPS   
14│siting_ConvNeXt-S│ImageNet: Architectures, Training and       │      │      │S +      │2023      
**│-ConvStem**}     │Generalization across Threat Models][225]*  │      │      │ConvStem │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**RodriguezMuno│*[Characterizing Model Robustness via       │77.76%│51.56%│Swin-B   │arXiv, Sep
15│z2024Characterizi│Natural Input Gradients][226]*              │      │      │         │2024      
**│ng_Swin-B**}     │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Singh2023Revi│*[Revisiting Adversarial Training for       │72.72%│49.46%│ConvNeXt-│NeurIPS   
16│siting_ConvNeXt-T│ImageNet: Architectures, Training and       │      │      │T +      │2023      
**│-ConvStem**}     │Generalization across Threat Models][227]*  │      │      │ConvStem │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Peng2023Robus│*[Robust Principles: Architectural Design   │73.44%│48.94%│RaWideRes│BMVC 2023 
17│t**}             │Principles for Adversarially Robust         │      │      │Net-101-2│          
**│                 │CNNs][228]*                                 │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Singh2023Revi│*[Revisiting Adversarial Training for       │72.56%│48.08%│ViT-S +  │NeurIPS   
18│siting_ViT-S-Conv│ImageNet: Architectures, Training and       │      │      │ConvStem │2023      
**│Stem**}          │Generalization across Threat Models][229]*  │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Debenedetti20│*[A Light Recipe to Train Robust Vision     │73.76%│47.60%│XCiT-L12 │arXiv, Sep
19│22Light_XCiT-L12*│Transformers][230]*                         │      │      │         │2022      
**│*}               │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Debenedetti20│*[A Light Recipe to Train Robust Vision     │74.04%│45.24%│XCiT-M12 │arXiv, Sep
20│22Light_XCiT-M12*│Transformers][231]*                         │      │      │         │2022      
**│*}               │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Debenedetti20│*[A Light Recipe to Train Robust Vision     │72.34%│41.78%│XCiT-S12 │arXiv, Sep
21│22Light_XCiT-S12*│Transformers][232]*                         │      │      │         │2022      
**│*}               │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Chen2024Data_│*[Data filtering for efficient adversarial  │68.76%│40.60%│WideResNe│Pattern   
22│WRN_50_2**}      │training][233]*                             │      │      │t-50-2   │Recognitio
**│                 │                                            │      │      │         │n 2024    
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Mo2022When_Sw│*[When Adversarial Training Meets Vision    │74.66%│38.30%│Swin-B   │NeurIPS   
23│in-B**}          │Transformers: Recipes from Training to      │      │      │         │2022      
**│                 │Architecture][234]*                         │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Salman2020Do_│*[Do Adversarially Robust ImageNet Models   │68.46%│38.14%│WideResNe│NeurIPS   
24│50_2**}          │Transfer Better?][235]*                     │      │      │t-50-2   │2020      
**│                 │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Salman2020Do_│*[Do Adversarially Robust ImageNet Models   │64.02%│34.96%│ResNet-50│NeurIPS   
25│R50**}           │Transfer Better?][236]*                     │      │      │         │2020      
**│                 │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Mo2022When_Vi│*[When Adversarial Training Meets Vision    │68.38%│34.40%│ViT-B    │NeurIPS   
26│T-B**}           │Transformers: Recipes from Training to      │      │      │         │2022      
**│                 │Architecture][237]*                         │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Engstrom2019R│*[Robustness library][238]*                 │62.56%│29.22%│ResNet-50│GitHub,   
27│obustness**}     │                                            │      │      │         │Oct 2019  
**│                 │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Wong2020Fast*│*[Fast is better than free: Revisiting      │55.62%│26.24%│ResNet-50│ICLR 2020 
28│*}               │adversarial training][239]*                 │      │      │         │          
**│                 │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Salman2020Do_│*[Do Adversarially Robust ImageNet Models   │52.92%│25.32%│ResNet-18│NeurIPS   
29│R18**}           │Transfer Better?][240]*                     │      │      │         │2020      
**│                 │                                            │      │      │         │          
──┼─────────────────┼────────────────────────────────────────────┼──────┼──────┼─────────┼──────────
**│^{**Standard_R50*│*[Standardly trained model][241]*           │76.52%│0.00% │ResNet-50│N/A       
30│*}               │                                            │      │      │         │          
**│                 │                                            │      │      │         │          
──┴─────────────────┴────────────────────────────────────────────┴──────┴──────┴─────────┴──────────

#### Corruptions (ImageNet-C & ImageNet-3DCC)

──┬────────────┬────────────────────────────────────────────────────┬───────┬───────┬───────┬───────
# │Model ID    │Paper                                               │Clean  │Robust │Archite│Venue  
  │            │                                                    │accurac│accurac│cture  │       
  │            │                                                    │y      │y      │       │       
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Tian2022│*[Deeper Insights into the Robustness of ViTs       │81.38% │67.55% │DeiT   │arXiv, 
1*│Deeper_DeiT-│towards Common Corruptions][242]*                   │       │       │Base   │Apr    
* │B**}        │                                                    │       │       │       │2022   
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Tian2022│*[Deeper Insights into the Robustness of ViTs       │79.76% │62.91% │DeiT   │arXiv, 
2*│Deeper_DeiT-│towards Common Corruptions][243]*                   │       │       │Small  │Apr    
* │S**}        │                                                    │       │       │       │2022   
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Erichson│*[NoisyMix: Boosting Robustness by Combining Data   │76.90% │53.28% │ResNet-│arXiv, 
3*│2022NoisyMix│Augmentations, Stability Training, and Noise        │       │       │50     │Feb    
* │_new**}     │Injections][244]*                                   │       │       │       │2022   
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Hendryck│*[The Many Faces of Robustness: A Critical Analysis │76.86% │52.90% │ResNet-│ICCV   
4*│s2020Many**}│of Out-of-Distribution Generalization][245]*        │       │       │50     │2021   
* │            │                                                    │       │       │       │       
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Erichson│*[NoisyMix: Boosting Robustness by Combining Data   │76.98% │52.47% │ResNet-│arXiv, 
5*│2022NoisyMix│Augmentations, Stability Training, and Noise        │       │       │50     │Feb    
* │**}         │Injections][246]*                                   │       │       │       │2022   
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Hendryck│*[AugMix: A Simple Data Processing Method to Improve│77.34% │49.33% │ResNet-│ICLR   
6*│s2020AugMix*│Robustness and Uncertainty][247]*                   │       │       │50     │2020   
* │*}          │                                                    │       │       │       │       
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Geirhos2│*[ImageNet-trained CNNs are biased towards texture; │74.98% │45.76% │ResNet-│ICLR   
7*│018_SIN_IN**│increasing shape bias improves accuracy and         │       │       │50     │2019   
* │}           │robustness][248]*                                   │       │       │       │       
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Geirhos2│*[ImageNet-trained CNNs are biased towards texture; │77.56% │42.00% │ResNet-│ICLR   
8*│018_SIN_IN_I│increasing shape bias improves accuracy and         │       │       │50     │2019   
* │N**}        │robustness][249]*                                   │       │       │       │       
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Geirhos2│*[ImageNet-trained CNNs are biased towards texture; │60.08% │39.92% │ResNet-│ICLR   
9*│018_SIN**}  │increasing shape bias improves accuracy and         │       │       │50     │2019   
* │            │robustness][250]*                                   │       │       │       │       
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Standard│*[Standardly trained model][251]*                   │76.72% │39.48% │ResNet-│N/A    
10│_R50**}     │                                                    │       │       │50     │       
**│            │                                                    │       │       │       │       
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**Salman20│*[Do Adversarially Robust ImageNet Models Transfer  │68.64% │36.09% │WideRes│NeurIPS
11│20Do_50_2_Li│Better?][252]*                                      │       │       │Net-50-│2020   
**│nf**}       │                                                    │       │       │2      │       
──┼────────────┼────────────────────────────────────────────────────┼───────┼───────┼───────┼───────
**│^{**AlexNet*│*[ImageNet Classification with Deep Convolutional   │56.24% │21.12% │AlexNet│NeurIPS
12│*}          │Neural Networks][253]*                              │       │       │       │2012   
**│            │                                                    │       │       │       │       
──┴────────────┴────────────────────────────────────────────────────┴───────┴───────┴───────┴───────

## Notebooks

We host all the notebooks at Google Colab:

* [RobustBench: quick start][254]: a quick tutorial to get started that illustrates the main
  features of **`RobustBench`**.
* [RobustBench: json stats][255]: various plots based on the jsons from `model_info` (robustness
  over venues, robustness vs accuracy, etc).

Feel free to suggest a new notebook based on the **Model Zoo** or the jsons from `model_info`. We
are very interested in collecting new insights about benefits and tradeoffs between different
perturbation types.

## How to contribute

Contributions to **`RobustBench`** are very welcome! You can help to improve **`RobustBench`**:

* Are you an author of a recent paper focusing on improving adversarial robustness? Consider adding
  new models (see the instructions below 👇).
* Do you have in mind some better *standardized* attack? Do you want to extend **`RobustBench`** to
  other threat models? We'll be glad to discuss that!
* Do you have an idea how to make the existing codebase better? Just open a pull request or create
  an issue and we'll be happy to discuss potential changes.

## Adding a new evaluation

In case you have some new (potentially, adaptive) evaluation that leads to a *lower* robust accuracy
than AutoAttack, we will be happy to add it to the leaderboard. The easiest way is to **open an
issue with the "New external evaluation(s)" template** and fill in all the fields.

## Adding a new model

#### Public model submission (Leaderboard + Model Zoo)

The easiest way to add new models to the leaderboard and/or to the model zoo, is by **opening an
issue with the "New Model(s)" template** and fill in all the fields.

In the following sections there are some tips on how to prepare the claim.

Claim

The claim can be computed in the following way (example for `cifar10`, `Linf` threat model):

import torch

from robustbench import benchmark
from myrobust model import MyRobustModel

threat_model = "Linf"  # one of {"Linf", "L2", "corruptions"}
dataset = "cifar10"  # one of {"cifar10", "cifar100", "imagenet"}

model = MyRobustModel()
model_name = "<Name><Year><FirstWordOfTheTitle>"
device = torch.device("cuda:0")

clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=10000, dataset=dataset,
                                  threat_model=threat_model, eps=8/255, device=device,
                                  to_disk=True)

In particular, the `to_disk` argument, if `True`, generates a json file at the path
`model_info/<dataset>/<threat_model>/<Name><Year><FirstWordOfTheTitle>.json` which is structured in
the following way (example from `model_info/cifar10/Linf/Rice2020Overfitting.json`):

{
  "link": "https://arxiv.org/abs/2002.11569",
  "name": "Overfitting in adversarially robust deep learning",
  "authors": "Leslie Rice, Eric Wong, J. Zico Kolter",
  "additional_data": false,
  "number_forward_passes": 1,
  "dataset": "cifar10",
  "venue": "ICML 2020",
  "architecture": "WideResNet-34-20",
  "eps": "8/255",
  "clean_acc": "85.34",
  "reported": "58",
  "autoattack_acc": "53.42"
}

The only difference is that the generated json will have only the fields `"clean_acc"` and
`"autoattack_acc"` (for `"Linf"` and `"L2"` threat models) or `"corruptions_acc"` (for the
`"corruptions"` threat model) already specified. The other fields have to be filled manually.

If the given `threat_model` is `corruptions`, we also save unaggregated results on the different
combinations of corruption types and severities in [this csv file][256] (for CIFAR-10).

For ImageNet benchmarks, the users should specify what preprocessing should be used (e.g. resize and
crop to the needed resolution). There are some preprocessings already defined in
[`robustbench.data.PREPROCESSINGS`][257], which can be used by specifying the key as the
`preprocessing` parameter of `benchmark`. Otherwise, it's possible to pass an arbitrary torchvision
transform (or torchvision-compatible transform), e.g.:

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=10000, dataset=dataset,
                                  threat_model=threat_model, eps=8/255, device=device,
                                  to_disk=True, preprocessing=transform)
Model definition

In case you want to add a model in the Model Zoo by yourself, then you should also open a PR with
the new model(s) you would like to add. All the models of each `<dataset>` are saved in
`robustbench/model_zoo/<dataset>.py`. Each file contains a dictionary for every threat model, where
the keys are the identifiers of each model, and the values are either class constructors, for models
that have to change standard architectures, or `lambda` functions that return the constructed model.

If your model is a standard architecture (e.g., `WideResNet`), does not apply any normalization to
the input nor has to do things differently from the standard architecture, consider adding your
model as a lambda function, e.g.

('Cui2020Learnable_34_10', {
    'model': lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
    'gdrive_id': '16s9pi_1QgMbFLISVvaVUiNfCzah6g2YV'
})

If your model is a standard architecture, but you need to do something differently (e.g. applying
normalization), consider inheriting the class defined in `wide_resnet.py` or `resnet.py`. For
example:

class Rice2020OverfittingNet(WideResNet):
    def __init__(self, depth, widen_factor):
        super(Rice2020OverfittingNet, self).__init__(depth=depth, widen_factor=widen_factor,
                                                     sub_block1=False)
        self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.2471, 0.2435, 0.2616]).float().view(3, 1, 1).cuda()

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Rice2020OverfittingNet, self).forward(x)

If instead you need to create a new architecture, please put it in
`robustbench/model_zoo/archietectures/<my_architecture>.py`.

Model checkpoint

You should also add your model entry in the corresponding `<threat_model>` dict in the file
`robustbench/model_zoo/<dataset>.py`. For instance, let's say your model is robust against common
corruptions in CIFAR-10 (i.e. CIFAR-10-C), then you should add your model to the
`common_corruptions` dict in [`robustbench/model_zoo/cifar10.py`][258].

The model should also contain the *Google Drive ID* with your PyTorch model so that it can be
downloaded automatically from Google Drive:

    ('Rice2020Overfitting', {
        'model': Rice2020OverfittingNet(34, 20),
        'gdrive_id': '1vC_Twazji7lBjeMQvAD9uEQxi9Nx2oG-',
})

#### Private model submission (leaderboard only)

In case you want to keep your checkpoints private for some reasons, you can also submit your claim
by opening an issue with the same "New Model(s)" template, specifying that the submission is
private, and sharing the checkpoints with the email address `adversarial.benchmark@gmail.com`. In
this case, we will add your model to the leaderboard but not to the Model Zoo and will not share
your checkpoints publicly.

#### License of the models

By default, the models are released under the MIT license, but you can also tell us if you want to
release your model under a customized license.

## Automatic tests

In order to run the tests, run:

* `python -m unittest discover tests -t . -v` for fast testing
* `RUN_SLOW=true python -m unittest discover tests -t . -v` for slower testing

For example, one can test if the clean accuracy on 200 examples exceeds some threshold (70%) or if
clean accuracy on 10'000 examples for each model matches the ones from the jsons located at
`robustbench/model_info`.

Note that one can specify some configurations like `batch_size`, `data_dir`, `model_dir` in
`tests/config.py` for running the tests.

## Citation

Would you like to reference the **`RobustBench`** leaderboard or you are using models from the
**Model Zoo**?
Then consider citing our [whitepaper][259]:

@inproceedings{croce2021robustbench,
  title     = {RobustBench: a standardized adversarial robustness benchmark},
  author    = {Croce, Francesco and Andriushchenko, Maksym and Sehwag, Vikash and Debenedetti, Edoar
do and Flammarion, Nicolas and Chiang, Mung and Mittal, Prateek and Matthias Hein},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchma
rks Track},
  year      = {2021},
  url       = {https://openreview.net/forum?id=SSKZPJCt7B}
}

## Contact

Feel free to contact us about anything related to **`RobustBench`** by creating an issue, a pull
request or by email at `adversarial.benchmark@gmail.com`.

[1]: https://robustbench.github.io/
[2]: https://arxiv.org/abs/2010.09670
[3]: https://3dcommoncorruptions.epfl.ch/
[4]: #new-evaluating-robustness-of-imagenet-models-against-3d-common-corruptions-imagenet-3dcc
[5]: https://github.com/RobustBench/robustbench/issues/59
[6]: https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html
[7]: https://arxiv.org/abs/1802.00420
[8]: https://arxiv.org/abs/2002.08347
[9]: https://arxiv.org/abs/1902.06705
[10]: https://arxiv.org/abs/1902.02918
[11]: https://github.com/fra31/auto-attack/blob/master/flags_doc.md
[12]: https://robustbench.github.io/
[13]: https://github.com/fra31/auto-attack
[14]: https://github.com/fra31/auto-attack
[15]: https://robustbench.github.io/
[16]: https://www.robust-ml.org/
[17]: https://www.robust-ml.org/
[18]: https://arxiv.org/abs/2002.08347
[19]: https://arxiv.org/abs/2007.08489
[20]: https://arxiv.org/abs/2007.05869
[21]: https://arxiv.org/abs/1805.12152
[22]: https://arxiv.org/abs/1910.08640
[23]: https://arxiv.org/abs/1906.00945
[24]: https://arxiv.org/abs/1811.03194
[25]: https://arxiv.org/abs/1906.07153
[26]: https://arxiv.org/abs/1911.09665
[27]: https://arxiv.org/abs/1909.11764
[28]: https://arxiv.org/abs/2004.10934
[29]: https://arxiv.org/abs/1911.09665
[30]: https://arxiv.org/abs/1905.01034
[31]: https://arxiv.org/abs/2008.03364
[32]: https://arxiv.org/abs/1906.06316
[33]: https://arxiv.org/abs/1902.08722
[34]: https://arxiv.org/abs/2009.04131
[35]: https://colab.research.google.com/drive/1MQY_7O9vj7ixD5ilVRbdQwlNPFvxifHV
[36]: https://arxiv.org/abs/1905.13736
[37]: https://github.com/bethgelab/foolbox
[38]: https://arxiv.org/abs/2003.01690
[39]: https://arxiv.org/abs/1901.10513
[40]: https://arxiv.org/abs/1906.08988
[41]: https://3dcommoncorruptions.epfl.ch/
[42]: https://github.com/EPFL-VILAB/3DCommonCorruptions#3dcc-data
[43]: https://image-net.org/download.php
[44]: https://github.com/soumith/imagenet-multiGPU.torch#data-processing
[45]: https://github.com/hendrycks/robustness#imagenet-c
[46]: https://github.com/EPFL-VILAB/3DCommonCorruptions#3dcc-data
[47]: https://robustbench.github.io/
[48]: https://arxiv.org/abs/2404.09349
[49]: https://arxiv.org/abs/2406.05927
[50]: https://arxiv.org/abs/2404.09349
[51]: https://arxiv.org/abs/2308.16258
[52]: https://arxiv.org/abs/2302.04638
[53]: https://arxiv.org/abs/2402.02263
[54]: https://arxiv.org/abs/2406.05927
[55]: https://arxiv.org/abs/2301.12554
[56]: https://arxiv.org/abs/2305.13948
[57]: https://arxiv.org/abs/2302.04638
[58]: https://arxiv.org/abs/2103.01946
[59]: https://arxiv.org/abs/2110.09468
[60]: https://arxiv.org/abs/2010.03593
[61]: https://arxiv.org/abs/2212.11005
[62]: https://arxiv.org/abs/2103.01946
[63]: https://arxiv.org/abs/2103.01946
[64]: https://arxiv.org/abs/2110.12976
[65]: https://arxiv.org/abs/2302.03015
[66]: https://arxiv.org/abs/2110.09468
[67]: https://arxiv.org/pdf/2202.10103.pdf
[68]: https://openreview.net/forum?id=BuD2LmNaU3a
[69]: https://arxiv.org/abs/2104.09425
[70]: https://arxiv.org/abs/2010.03593
[71]: https://arxiv.org/abs/2110.03825
[72]: https://arxiv.org/abs/2110.03825
[73]: https://arxiv.org/abs/2110.05626
[74]: https://arxiv.org/pdf/2202.10103.pdf
[75]: https://openreview.net/forum?id=BuD2LmNaU3a
[76]: https://arxiv.org/abs/2103.01946
[77]: https://arxiv.org/abs/2106.02078
[78]: https://arxiv.org/abs/2104.09425
[79]: https://arxiv.org/abs/2004.05884
[80]: https://arxiv.org/abs/2106.02078
[81]: https://arxiv.org/abs/2010.01736
[82]: https://arxiv.org/abs/1905.13736
[83]: https://arxiv.org/abs/2110.09468
[84]: https://doi.org/10.1016/j.patcog.2024.110394
[85]: https://arxiv.org/abs/2210.09852
[86]: https://arxiv.org/abs/2210.15318
[87]: https://arxiv.org/abs/2111.02331
[88]: https://openreview.net/forum?id=BuD2LmNaU3a
[89]: https://arxiv.org/abs/2203.06616
[90]: https://arxiv.org/abs/2209.07399
[91]: https://doi.org/10.1016/j.patcog.2024.110394
[92]: https://arxiv.org/abs/2209.07399
[93]: https://arxiv.org/abs/2002.10509
[94]: https://arxiv.org/abs/2010.03593
[95]: https://openreview.net/forum?id=BuD2LmNaU3a
[96]: https://arxiv.org/abs/2305.13948
[97]: https://arxiv.org/abs/2111.02331
[98]: https://arxiv.org/abs/2010.03593
[99]: https://arxiv.org/abs/2103.01946
[100]: https://openreview.net/forum?id=rklOg6EFwS
[101]: https://arxiv.org/abs/2203.06616
[102]: https://arxiv.org/abs/2004.05884
[103]: https://arxiv.org/abs/2209.07399
[104]: https://arxiv.org/abs/2104.09425
[105]: https://arxiv.org/abs/1901.09960
[106]: https://arxiv.org/abs/2002.08619
[107]: https://arxiv.org/abs/2011.11164
[108]: https://arxiv.org/abs/2002.11242
[109]: https://arxiv.org/abs/2002.11569
[110]: https://arxiv.org/abs/2002.10319
[111]: https://arxiv.org/abs/1901.08573
[112]: https://arxiv.org/abs/2011.11164
[113]: https://arxiv.org/abs/2210.15318
[114]: https://arxiv.org/abs/2003.12862
[115]: https://arxiv.org/abs/2010.01278
[116]: https://arxiv.org/abs/2210.09852
[117]: https://arxiv.org/abs/2003.09347
[118]: https://github.com/MadryLab/robustness
[119]: https://arxiv.org/abs/1905.00877
[120]: https://arxiv.org/abs/2007.02617
[121]: https://arxiv.org/abs/2001.03994
[122]: https://openreview.net/forum?id=HkeryxBtPB
[123]: https://github.com/RobustBench/robustbench/
[124]: https://arxiv.org/abs/2302.04638
[125]: https://arxiv.org/abs/2406.05927
[126]: https://arxiv.org/abs/2302.04638
[127]: https://arxiv.org/abs/2103.01946
[128]: https://arxiv.org/abs/2010.03593
[129]: https://arxiv.org/abs/2103.01946
[130]: https://arxiv.org/abs/2103.01946
[131]: https://arxiv.org/abs/2003.09461
[132]: https://arxiv.org/abs/2104.09425
[133]: https://arxiv.org/abs/2003.09461
[134]: https://openreview.net/forum?id=BuD2LmNaU3a
[135]: https://arxiv.org/abs/2103.01946
[136]: https://arxiv.org/abs/2010.03593
[137]: https://arxiv.org/abs/2104.09425
[138]: https://arxiv.org/abs/2004.05884
[139]: https://arxiv.org/abs/2003.09461
[140]: https://github.com/MadryLab/robustness
[141]: https://arxiv.org/abs/2002.11569
[142]: https://arxiv.org/abs/1811.09600
[143]: https://openreview.net/forum?id=HkeryxBtPB
[144]: https://github.com/RobustBench/robustbench/
[145]: https://arxiv.org/abs/2106.09129
[146]: https://arxiv.org/abs/2106.09129
[147]: https://arxiv.org/abs/2106.09129
[148]: https://arxiv.org/abs/2103.02325
[149]: https://arxiv.org/abs/1912.02781
[150]: https://arxiv.org/abs/2112.13547
[151]: https://arxiv.org/abs/1912.02781
[152]: https://arxiv.org/abs/2103.02325
[153]: https://arxiv.org/abs/2106.09129
[154]: https://arxiv.org/abs/2103.01946
[155]: https://arxiv.org/abs/2103.02325
[156]: https://arxiv.org/abs/2103.02325
[157]: https://arxiv.org/abs/2103.02325
[158]: https://arxiv.org/abs/2103.01946
[159]: https://artofrobust.github.io/short_paper/31.pdf
[160]: https://openreview.net/forum?id=SHB_znlW5G7
[161]: https://github.com/RobustBench/robustbench/
[162]: https://arxiv.org/abs/2302.04638
[163]: https://arxiv.org/abs/2406.05927
[164]: https://arxiv.org/abs/2402.02263
[165]: https://arxiv.org/abs/2305.13948
[166]: https://arxiv.org/abs/2302.04638
[167]: https://arxiv.org/abs/2301.12554
[168]: https://arxiv.org/abs/2010.03593
[169]: https://arxiv.org/abs/2301.12554
[170]: https://arxiv.org/abs/2209.07399
[171]: https://arxiv.org/abs/2103.01946
[172]: https://arxiv.org/abs/2209.07399
[173]: https://arxiv.org/pdf/2202.10103.pdf
[174]: https://arxiv.org/abs/2305.13948
[175]: https://arxiv.org/abs/2209.07399
[176]: https://arxiv.org/abs/2103.01946
[177]: https://arxiv.org/abs/2203.06616
[178]: https://arxiv.org/abs/2305.13948
[179]: https://arxiv.org/abs/2210.15318
[180]: https://arxiv.org/abs/2011.11164
[181]: https://arxiv.org/abs/2104.09425
[182]: https://doi.org/10.1016/j.patcog.2024.110394
[183]: https://arxiv.org/pdf/2202.10103.pdf
[184]: https://arxiv.org/abs/2203.06616
[185]: https://arxiv.org/abs/2111.02331
[186]: https://arxiv.org/abs/2210.09852
[187]: https://arxiv.org/abs/2011.11164
[188]: https://arxiv.org/abs/2010.03593
[189]: https://arxiv.org/abs/2011.11164
[190]: https://openreview.net/forum?id=BuD2LmNaU3a
[191]: https://arxiv.org/abs/2004.05884
[192]: https://arxiv.org/abs/2103.01946
[193]: https://arxiv.org/abs/1901.09960
[194]: https://arxiv.org/abs/2210.15318
[195]: https://arxiv.org/abs/2011.11164
[196]: https://arxiv.org/abs/2210.09852
[197]: https://arxiv.org/abs/2010.01278
[198]: https://arxiv.org/abs/2003.09347
[199]: https://arxiv.org/abs/2002.11569
[200]: https://arxiv.org/abs/2106.09129
[201]: https://arxiv.org/abs/2106.09129
[202]: https://arxiv.org/abs/2112.13547
[203]: https://arxiv.org/abs/2106.09129
[204]: https://arxiv.org/abs/2106.09129
[205]: https://arxiv.org/abs/1912.02781
[206]: https://arxiv.org/abs/1912.02781
[207]: https://artofrobust.github.io/short_paper/31.pdf
[208]: https://arxiv.org/abs/2010.03593
[209]: https://openreview.net/forum?id=SHB_znlW5G7
[210]: https://openreview.net/forum?id=SHB_znlW5G7
[211]: https://arxiv.org/abs/2010.03593
[212]: https://arxiv.org/abs/2312.04960
[213]: https://arxiv.org/abs/2302.14301
[214]: https://arxiv.org/abs/2406.05927
[215]: https://arxiv.org/abs/2402.02263
[216]: https://arxiv.org/abs/2302.14301
[217]: https://arxiv.org/abs/2406.05927
[218]: https://arxiv.org/abs/2303.01870
[219]: https://arxiv.org/abs/2302.14301
[220]: https://arxiv.org/abs/2303.01870
[221]: https://arxiv.org/abs/2312.04960
[222]: https://arxiv.org/abs/2302.14301
[223]: https://arxiv.org/abs/2303.01870
[224]: https://arxiv.org/abs/2409.20139
[225]: https://arxiv.org/abs/2303.01870
[226]: https://arxiv.org/abs/2409.20139
[227]: https://arxiv.org/abs/2303.01870
[228]: https://arxiv.org/abs/2308.16258
[229]: https://arxiv.org/abs/2303.01870
[230]: https://arxiv.org/abs/2209.07399
[231]: https://arxiv.org/abs/2209.07399
[232]: https://arxiv.org/abs/2209.07399
[233]: https://doi.org/10.1016/j.patcog.2024.110394
[234]: https://arxiv.org/abs/2210.07540
[235]: https://arxiv.org/abs/2007.08489
[236]: https://arxiv.org/abs/2007.08489
[237]: https://arxiv.org/abs/2210.07540
[238]: https://github.com/MadryLab/robustness
[239]: https://arxiv.org/abs/2001.03994
[240]: https://arxiv.org/abs/2007.08489
[241]: https://github.com/RobustBench/robustbench/
[242]: https://arxiv.org/abs/2204.12143
[243]: https://arxiv.org/abs/2204.12143
[244]: https://arxiv.org/pdf/2202.01263.pdf
[245]: https://arxiv.org/abs/2006.16241
[246]: https://arxiv.org/pdf/2202.01263.pdf
[247]: https://arxiv.org/abs/1912.02781
[248]: https://arxiv.org/abs/1811.12231
[249]: https://arxiv.org/abs/1811.12231
[250]: https://arxiv.org/abs/1811.12231
[251]: https://github.com/RobustBench/robustbench/
[252]: https://arxiv.org/abs/2007.08489
[253]: https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
[254]: https://colab.research.google.com/drive/1MQY_7O9vj7ixD5ilVRbdQwlNPFvxifHV
[255]: https://colab.research.google.com/drive/19tgblr13SvaCpG8hoOTv6QCULVJbCec6
[256]: /RobustBench/robustbench/blob/master/model_info/cifar10/corruptions/unaggregated_results.csv
[257]: https://github.com/RobustBench/robustbench/blob/imagenet-preprocessing/robustbench/data.py#L1
8
[258]: /RobustBench/robustbench/blob/master/robustbench/model_zoo/cifar10.py
[259]: https://arxiv.org/abs/2010.09670
