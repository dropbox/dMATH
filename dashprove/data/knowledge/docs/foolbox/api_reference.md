* »
* `foolbox.attacks`
* [ Edit on GitHub][1]

# [`foolbox.attacks`][2][][3]

───────────────────────────┬────────────────────────────────────────────────────────────────────────
[`L2ContrastReductionAttack│Reduces the contrast of the input using a perturbation of the given size
`][4]                      │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`VirtualAdversarialAttack`│Second-order gradient-based attack on the logits.                       
][5]                       │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`DDNAttack`][6]           │The Decoupled Direction and Norm L2 adversarial attack.                 
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2ProjectedGradientDescen│L2 Projected Gradient Descent                                           
tAttack`][7]               │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinfProjectedGradientDesc│Linf Projected Gradient Descent                                         
entAttack`][8]             │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2BasicIterativeAttack`][│L2 Basic Iterative Method                                               
9]                         │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinfBasicIterativeAttack`│L-infinity Basic Iterative Method                                       
][10]                      │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2FastGradientAttack`][11│Fast Gradient Method (FGM)                                              
]                          │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinfFastGradientAttack`][│Fast Gradient Sign Method (FGSM)                                        
12]                        │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2AdditiveGaussianNoiseAt│Samples Gaussian noise with a fixed L2 size.                            
tack`][13]                 │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2AdditiveUniformNoiseAtt│Samples uniform noise with a fixed L2 size.                             
ack`][14]                  │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2ClippingAwareAdditiveGa│Samples Gaussian noise with a fixed L2 size after clipping.             
ussianNoiseAttack`][15]    │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2ClippingAwareAdditiveUn│Samples uniform noise with a fixed L2 size after clipping.              
iformNoiseAttack`][16]     │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinfAdditiveUniformNoiseA│Samples uniform noise with a fixed L-infinity size                      
ttack`][17]                │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2RepeatedAdditiveGaussia│Repeatedly samples Gaussian noise with a fixed L2 size.                 
nNoiseAttack`][18]         │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2RepeatedAdditiveUniform│Repeatedly samples uniform noise with a fixed L2 size.                  
NoiseAttack`][19]          │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2ClippingAwareRepeatedAd│Repeatedly samples Gaussian noise with a fixed L2 size after clipping.  
ditiveGaussianNoiseAttack`]│                                                                        
[20]                       │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2ClippingAwareRepeatedAd│Repeatedly samples uniform noise with a fixed L2 size after clipping.   
ditiveUniformNoiseAttack`][│                                                                        
21]                        │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinfRepeatedAdditiveUnifo│Repeatedly samples uniform noise with a fixed L-infinity size.          
rmNoiseAttack`][22]        │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`InversionAttack`][23]    │Creates "negative images" by inverting the pixel values.                
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`BinarySearchContrastReduc│Reduces the contrast of the input using a binary search to find the     
tionAttack`][24]           │smallest adversarial perturbation                                       
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinearSearchContrastReduc│Reduces the contrast of the input using a linear search to find the     
tionAttack`][25]           │smallest adversarial perturbation                                       
───────────────────────────┼────────────────────────────────────────────────────────────────────────
`HopSkipJumpAttack`        │A powerful adversarial attack that requires neither gradients nor       
                           │probabilities [#Chen19].                                                
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2CarliniWagnerAttack`][2│Implementation of the Carlini & Wagner L2 Attack.                       
6]                         │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`NewtonFoolAttack`][27]   │Implementation of the NewtonFool Attack.                                
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`EADAttack`][28]          │Implementation of the EAD Attack with EN Decision Rule.                 
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`GaussianBlurAttack`][29] │Blurs the inputs using a Gaussian filter with linearly increasing       
                           │standard deviation.                                                     
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2DeepFoolAttack`][30]   │A simple and fast gradient-based adversarial attack.                    
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinfDeepFoolAttack`][31] │A simple and fast gradient-based adversarial attack.                    
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`SaltAndPepperNoiseAttack`│Increases the amount of salt and pepper noise until the input is        
][32]                      │misclassified.                                                          
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinearSearchBlendedUnifor│Blends the input with a uniform noise input until it is misclassified.  
mNoiseAttack`][33]         │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`BinarizationRefinementAtt│For models that preprocess their inputs by binarizing the inputs, this  
ack`][34]                  │attack can improve adversarials found by other attacks.                 
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`DatasetAttack`][35]      │Draws randomly from the given dataset until adversarial examples for all
                           │inputs have been found.                                                 
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`BoundaryAttack`][36]     │A powerful adversarial attack that requires neither gradients nor       
                           │probabilities.                                                          
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L0BrendelBethgeAttack`][3│L0 variant of the Brendel & Bethge adversarial attack.                  
7]                         │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L1BrendelBethgeAttack`][3│L1 variant of the Brendel & Bethge adversarial attack.                  
8]                         │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2BrendelBethgeAttack`][3│L2 variant of the Brendel & Bethge adversarial attack.                  
9]                         │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinfinityBrendelBethgeAtt│L-infinity variant of the Brendel & Bethge adversarial attack.          
ack`][40]                  │                                                                        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L0FMNAttack`][41]        │The L0 Fast Minimum Norm adversarial attack, in Lp norm.                
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L1FMNAttack`][42]        │The L1 Fast Minimum Norm adversarial attack, in Lp norm.                
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2FMNAttack`][43]        │The L2 Fast Minimum Norm adversarial attack, in Lp norm.                
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LInfFMNAttack`][44]      │The L-infinity Fast Minimum Norm adversarial attack, in Lp norm.        
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`PointwiseAttack`][45]    │Starts with an adversarial and performs a binary search between the     
                           │adversarial and the original for each dimension of the input            
                           │individually.                                                           
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`FGM`][46]                │alias of                                                                
                           │[`foolbox.attacks.fast_gradient_method.L2FastGradientAttack`][47]       
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`FGSM`][48]               │alias of                                                                
                           │[`foolbox.attacks.fast_gradient_method.LinfFastGradientAttack`][49]     
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`L2PGD`][50]              │alias of                                                                
                           │[`foolbox.attacks.projected_gradient_descent.L2ProjectedGradientDescentA
                           │ttack`][51]                                                             
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`LinfPGD`][52]            │alias of                                                                
                           │[`foolbox.attacks.projected_gradient_descent.LinfProjectedGradientDescen
                           │tAttack`][53]                                                           
───────────────────────────┼────────────────────────────────────────────────────────────────────────
[`PGD`][54]                │alias of                                                                
                           │[`foolbox.attacks.projected_gradient_descent.LinfProjectedGradientDescen
                           │tAttack`][55]                                                           
───────────────────────────┴────────────────────────────────────────────────────────────────────────

* *class *foolbox.attacks.L2ContrastReductionAttack(***, *target=0.5*)[][56]*
  Reduces the contrast of the input using a perturbation of the given size
  
  *Parameters*
    **target** (*float*) – Target relative to the bounds from 0 (min) to 1 (max) towards which the
    contrast is reduced

* *class *foolbox.attacks.VirtualAdversarialAttack(*steps*, *xi=1e-06*)[][57]*
  Second-order gradient-based attack on the logits. [1][58] The attack calculate an untargeted
  adversarial perturbation by performing a approximated second order optimization step on the KL
  divergence between the unperturbed predictions and the predictions for the adversarial
  perturbation. This attack was originally introduced as the Virtual Adversarial Training [1][59]
  method.
  
  *Parameters*
    * **steps** (*int*) – Number of update steps.
    * **xi** (*float*) – L2 distance between original image and first adversarial proposal.
  
  References
  
  *1([1][60],[2][61])*
    Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae, Shin Ishii, “Distributional
    Smoothing with Virtual Adversarial Training”, [https://arxiv.org/abs/1507.00677][62]

* *class *foolbox.attacks.DDNAttack(***, *init_epsilon=1.0*, *steps=100*, *gamma=0.05*)[][63]*
  The Decoupled Direction and Norm L2 adversarial attack. [2][64]
  
  *Parameters*
    * **init_epsilon** (*float*) – Initial value for the norm/epsilon ball.
    * **steps** (*int*) – Number of steps for the optimization.
    * **gamma** (*float*) – Factor by which the norm will be modified: new_norm = norm * (1 + or -
      gamma).
  
  References
  
  *[2][65]*
    Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed, Robert Sabourin, Eric Granger,
    “Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and
    Defenses”, [https://arxiv.org/abs/1811.09600][66]

* *class *foolbox.attacks.L2ProjectedGradientDescentAttack(***, *rel_stepsize=0.025*,
*abs_stepsize=None*, *steps=50*, *random_start=True*)[][67]*
  L2 Projected Gradient Descent
  
  *Parameters*
    * **rel_stepsize** (*float*) – Stepsize relative to epsilon
    * **abs_stepsize** (*Optional**[**float**]*) – If given, it takes precedence over rel_stepsize.
    * **steps** (*int*) – Number of update steps to perform.
    * **random_start** (*bool*) – Whether the perturbation is initialized randomly or starts at
      zero.

* *class *foolbox.attacks.LinfProjectedGradientDescentAttack(***,
*rel_stepsize=0.03333333333333333*, *abs_stepsize=None*, *steps=40*, *random_start=True*)[][68]*
  Linf Projected Gradient Descent
  
  *Parameters*
    * **rel_stepsize** (*float*) – Stepsize relative to epsilon (defaults to 0.01 / 0.3).
    * **abs_stepsize** (*Optional**[**float**]*) – If given, it takes precedence over rel_stepsize.
    * **steps** (*int*) – Number of update steps to perform.
    * **random_start** (*bool*) – Whether the perturbation is initialized randomly or starts at
      zero.

* *class *foolbox.attacks.L2BasicIterativeAttack(***, *rel_stepsize=0.2*, *abs_stepsize=None*,
*steps=10*, *random_start=False*)[][69]*
  L2 Basic Iterative Method
  
  *Parameters*
    * **rel_stepsize** (*float*) – Stepsize relative to epsilon.
    * **abs_stepsize** (*Optional**[**float**]*) – If given, it takes precedence over rel_stepsize.
    * **steps** (*int*) – Number of update steps.
    * **random_start** (*bool*) – Controls whether to randomly start within allowed epsilon ball.

* *class *foolbox.attacks.LinfBasicIterativeAttack(***, *rel_stepsize=0.2*, *abs_stepsize=None*,
*steps=10*, *random_start=False*)[][70]*
  L-infinity Basic Iterative Method
  
  *Parameters*
    * **rel_stepsize** (*float*) – Stepsize relative to epsilon.
    * **abs_stepsize** (*Optional**[**float**]*) – If given, it takes precedence over rel_stepsize.
    * **steps** (*int*) – Number of update steps.
    * **random_start** (*bool*) – Controls whether to randomly start within allowed epsilon ball.

* *class *foolbox.attacks.L2FastGradientAttack(***, *random_start=False*)[][71]*
  Fast Gradient Method (FGM)
  
  *Parameters*
    **random_start** (*bool*) – Controls whether to randomly start within allowed epsilon ball.

* *class *foolbox.attacks.LinfFastGradientAttack(***, *random_start=False*)[][72]*
  Fast Gradient Sign Method (FGSM)
  
  *Parameters*
    **random_start** (*bool*) – Controls whether to randomly start within allowed epsilon ball.

* *class *foolbox.attacks.L2AdditiveGaussianNoiseAttack[][73]*
  Samples Gaussian noise with a fixed L2 size.

* *class *foolbox.attacks.L2AdditiveUniformNoiseAttack[][74]*
  Samples uniform noise with a fixed L2 size.

* *class *foolbox.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack[][75]*
  Samples Gaussian noise with a fixed L2 size after clipping.
  
  The implementation is based on [[#Rauber20]_][76].
  
  References
  
  *3*
    Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware Normalization and Rescaling”
    [https://arxiv.org/abs/2007.07677][77]

* *class *foolbox.attacks.L2ClippingAwareAdditiveUniformNoiseAttack[][78]*
  Samples uniform noise with a fixed L2 size after clipping.
  
  The implementation is based on [[#Rauber20]_][79].
  
  References
  
  *4*
    Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware Normalization and Rescaling”
    [https://arxiv.org/abs/2007.07677][80]

* *class *foolbox.attacks.LinfAdditiveUniformNoiseAttack[][81]*
  Samples uniform noise with a fixed L-infinity size

* *class *foolbox.attacks.L2RepeatedAdditiveGaussianNoiseAttack(***, *repeats=100*,
*check_trivial=True*)[][82]*
  Repeatedly samples Gaussian noise with a fixed L2 size.
  
  *Parameters*
    * **repeats** (*int*) – How often to sample random noise.
    * **check_trivial** (*bool*) – Check whether original sample is already adversarial.

* *class *foolbox.attacks.L2RepeatedAdditiveUniformNoiseAttack(***, *repeats=100*,
*check_trivial=True*)[][83]*
  Repeatedly samples uniform noise with a fixed L2 size.
  
  *Parameters*
    * **repeats** (*int*) – How often to sample random noise.
    * **check_trivial** (*bool*) – Check whether original sample is already adversarial.

* *class *foolbox.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(***, *repeats=100*,
*check_trivial=True*)[][84]*
  Repeatedly samples Gaussian noise with a fixed L2 size after clipping.
  
  The implementation is based on [[#Rauber20]_][85].
  
  References
  
  *5*
    Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware Normalization and Rescaling”
    [https://arxiv.org/abs/2007.07677][86]
  
  *Parameters*
    * **repeats** (*int*) – How often to sample random noise.
    * **check_trivial** (*bool*) – Check whether original sample is already adversarial.

* *class *foolbox.attacks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(***, *repeats=100*,
*check_trivial=True*)[][87]*
  Repeatedly samples uniform noise with a fixed L2 size after clipping.
  
  The implementation is based on [[#Rauber20]_][88].
  
  References
  
  *6*
    Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware Normalization and Rescaling”
    [https://arxiv.org/abs/2007.07677][89]
  
  *Parameters*
    * **repeats** (*int*) – How often to sample random noise.
    * **check_trivial** (*bool*) – Check whether original sample is already adversarial.

* *class *foolbox.attacks.LinfRepeatedAdditiveUniformNoiseAttack(***, *repeats=100*,
*check_trivial=True*)[][90]*
  Repeatedly samples uniform noise with a fixed L-infinity size.
  
  *Parameters*
    * **repeats** (*int*) – How often to sample random noise.
    * **check_trivial** (*bool*) – Check whether original sample is already adversarial.

* *class *foolbox.attacks.InversionAttack(***, *distance=None*)[][91]*
  Creates “negative images” by inverting the pixel values. [7][92]
  
  References
  
  *[7][93]*
    Hossein Hosseini, Baicen Xiao, Mayoore Jaiswal, Radha Poovendran, “On the Limitation of
    Convolutional Neural Networks in Recognizing Negative Images”,
    [https://arxiv.org/abs/1607.02533][94]
  
  *Parameters*
    **distance** (*Optional**[*[*foolbox.distances.Distance*][95]*]*) –

* *class *foolbox.attacks.BinarySearchContrastReductionAttack(***, *distance=None*,
*binary_search_steps=15*, *target=0.5*)[][96]*
  Reduces the contrast of the input using a binary search to find the smallest adversarial
  perturbation
  
  *Parameters*
    * **distance** (*Optional**[*[*foolbox.distances.Distance*][97]*]*) – Distance measure for which
      minimal adversarial examples are searched.
    * **binary_search_steps** (*int*) – Number of iterations in the binary search. This controls the
      precision of the results.
    * **target** (*float*) – Target relative to the bounds from 0 (min) to 1 (max) towards which the
      contrast is reduced

* *class *foolbox.attacks.LinearSearchContrastReductionAttack(***, *distance=None*, *steps=1000*,
*target=0.5*)[][98]*
  Reduces the contrast of the input using a linear search to find the smallest adversarial
  perturbation
  
  *Parameters*
    * **distance** (*Optional**[*[*foolbox.distances.Distance*][99]*]*) –
    * **steps** (*int*) –
    * **target** (*float*) –

* *class *foolbox.attacks.L2CarliniWagnerAttack(*binary_search_steps=9*, *steps=10000*,
*stepsize=0.01*, *confidence=0*, *initial_const=0.001*, *abort_early=True*)[][100]*
  Implementation of the Carlini & Wagner L2 Attack. [8][101]
  
  *Parameters*
    * **binary_search_steps** (*int*) – Number of steps to perform in the binary search over the
      const c.
    * **steps** (*int*) – Number of optimization steps within each binary search step.
    * **stepsize** (*float*) – Stepsize to update the examples.
    * **confidence** (*float*) – Confidence required for an example to be marked as adversarial.
      Controls the gap between example and decision boundary.
    * **initial_const** (*float*) – Initial value of the const c with which the binary search
      starts.
    * **abort_early** (*bool*) – Stop inner search as soons as an adversarial example has been
      found. Does not affect the binary search over the const c.
  
  References
  
  *[8][102]*
    Nicholas Carlini, David Wagner, “Towards evaluating the robustness of neural networks. In 2017
    ieee symposium on security and privacy” [https://arxiv.org/abs/1608.04644][103]

* *class *foolbox.attacks.NewtonFoolAttack(*steps=100*, *stepsize=0.01*)[][104]*
  Implementation of the NewtonFool Attack. [9][105]
  
  *Parameters*
    * **steps** (*int*) – Number of update steps to perform.
    * **step_size** – Size of each update step.
    * **stepsize** (*float*) –
  
  References
  
  *[9][106]*
    Uyeong Jang et al., “Objective Metrics and Gradient Descent Algorithms for Adversarial Examples
    in Machine Learning”, [https://dl.acm.org/citation.cfm?id=3134635][107]

* *class *foolbox.attacks.EADAttack(*binary_search_steps=9*, *steps=10000*, *initial_stepsize=0.01*,
*confidence=0.0*, *initial_const=0.001*, *regularization=0.01*, *decision_rule='EN'*,
*abort_early=True*)[][108]*
  Implementation of the EAD Attack with EN Decision Rule. [10][109]
  
  *Parameters*
    * **binary_search_steps** (*int*) – Number of steps to perform in the binary search over the
      const c.
    * **steps** (*int*) – Number of optimization steps within each binary search step.
    * **initial_stepsize** (*float*) – Initial stepsize to update the examples.
    * **confidence** (*float*) – Confidence required for an example to be marked as adversarial.
      Controls the gap between example and decision boundary.
    * **initial_const** (*float*) – Initial value of the const c with which the binary search
      starts.
    * **regularization** (*float*) – Controls the L1 regularization.
    * **decision_rule** (*Union**[**typing_extensions.Literal**[**'EN'**]**,
      **typing_extensions.Literal**[**'L1'**]**]*) – Rule according to which the best adversarial
      examples are selected. They either minimize the L1 or ElasticNet distance.
    * **abort_early** (*bool*) – Stop inner search as soons as an adversarial example has been
      found. Does not affect the binary search over the const c.
  
  References
  
  *[10][110]*
    Pin-Yu Chen, Yash Sharma, Huan Zhang, Jinfeng Yi, Cho-Jui Hsieh,
  
  “EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples”,
  [https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16893][111]

* *class *foolbox.attacks.GaussianBlurAttack(***, *distance=None*, *steps=1000*,
*channel_axis=None*, *max_sigma=None*)[][112]*
  Blurs the inputs using a Gaussian filter with linearly increasing standard deviation.
  
  *Parameters*
    * **steps** (*int*) – Number of sigma values tested between 0 and max_sigma.
    * **channel_axis** (*Optional**[**int**]*) – Index of the channel axis in the input data.
    * **max_sigma** (*Optional**[**float**]*) – Maximally allowed sigma value of the Gaussian blur.
    * **distance** (*Optional**[*[*foolbox.distances.Distance*][113]*]*) –

* *class *foolbox.attacks.L2DeepFoolAttack(***, *steps=50*, *candidates=10*, *overshoot=0.02*,
*loss='logits'*)[][114]*
  A simple and fast gradient-based adversarial attack.
  
  Implements the DeepFool L2 attack. [11][115]
  
  *Parameters*
    * **steps** (*int*) – Maximum number of steps to perform.
    * **candidates** (*Optional**[**int**]*) – Limit on the number of the most likely classes that
      should be considered. A small value is usually sufficient and much faster.
    * **overshoot** (*float*) – How much to overshoot the boundary.
    * **function.** (*loss Loss function to use inside the update*) –
    * **loss** (*Union**[**typing_extensions.Literal**[**'logits'**]**,
      **typing_extensions.Literal**[**'crossentropy'**]**]*) –
  
  References
  
  *[11][116]*
    Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, “DeepFool: a simple and
    accurate method to fool deep neural networks”, [https://arxiv.org/abs/1511.04599][117]

* *class *foolbox.attacks.LinfDeepFoolAttack(***, *steps=50*, *candidates=10*, *overshoot=0.02*,
*loss='logits'*)[][118]*
  A simple and fast gradient-based adversarial attack.
  
  Implements the [DeepFool][119] L-Infinity attack.
  
  *Parameters*
    * **steps** (*int*) – Maximum number of steps to perform.
    * **candidates** (*Optional**[**int**]*) – Limit on the number of the most likely classes that
      should be considered. A small value is usually sufficient and much faster.
    * **overshoot** (*float*) – How much to overshoot the boundary.
    * **function.** (*loss Loss function to use inside the update*) –
    * **loss** (*Union**[**typing_extensions.Literal**[**'logits'**]**,
      **typing_extensions.Literal**[**'crossentropy'**]**]*) –

* *class *foolbox.attacks.SaltAndPepperNoiseAttack(*steps=1000*, *across_channels=True*,
*channel_axis=None*)[][120]*
  Increases the amount of salt and pepper noise until the input is misclassified.
  
  *Parameters*
    * **steps** (*int*) – The number of steps to run.
    * **across_channels** (*bool*) – Whether the noise should be the same across all channels.
    * **channel_axis** (*Optional**[**int**]*) – The axis across which the noise should be the same
      (if across_channels is True). If None, will be automatically inferred from the model if
      possible.

* *class *foolbox.attacks.LinearSearchBlendedUniformNoiseAttack(***, *distance=None*,
*directions=1000*, *steps=1000*)[][121]*
  Blends the input with a uniform noise input until it is misclassified.
  
  *Parameters*
    * **distance** (*Optional**[*[*foolbox.distances.Distance*][122]*]*) – Distance measure for
      which minimal adversarial examples are searched.
    * **directions** (*int*) – Number of random directions in which the perturbation is searched.
    * **steps** (*int*) – Number of blending steps between the original image and the random
      directions.

* *class *foolbox.attacks.BinarizationRefinementAttack(***, *distance=None*, *threshold=None*,
*included_in='upper'*)[][123]*
  For models that preprocess their inputs by binarizing the inputs, this attack can improve
  adversarials found by other attacks. It does this by utilizing information about the binarization
  and mapping values to the corresponding value in the clean input or to the right side of the
  threshold.
  
  *Parameters*
    * **threshold** (*Optional**[**float**]*) – The threshold used by the models binarization. If
      none, defaults to (model.bounds()[1] - model.bounds()[0]) / 2.
    * **included_in** (*Union**[**typing_extensions.Literal**[**'lower'**]**,
      **typing_extensions.Literal**[**'upper'**]**]*) – Whether the threshold value itself belongs
      to the lower or upper interval.
    * **distance** (*Optional**[*[*foolbox.distances.Distance*][124]*]*) –

* *class *foolbox.attacks.DatasetAttack(***, *distance=None*)[][125]*
  Draws randomly from the given dataset until adversarial examples for all inputs have been found.
  
  To pass data form the dataset to this attack, call `feed()`. `feed()` can be called several times
  and should only be called with batches that are small enough that they can be passed through the
  model.
  
  *Parameters*
    **distance** (*Optional**[*[*foolbox.distances.Distance*][126]*]*) – Distance measure for which
    minimal adversarial examples are searched.

* *class *foolbox.attacks.BoundaryAttack(*init_attack=None*, *steps=25000*, *spherical_step=0.01*,
*source_step=0.01*, *source_step_convergance=1e-07*, *step_adaptation=1.5*, *tensorboard=False*,
*update_stats_every_k=10*)[][127]*
  A powerful adversarial attack that requires neither gradients nor probabilities.
  
  This is the reference implementation for the attack. [12][128]
  
  Notes
  
  Differences to the original reference implementation: * We do not perform internal operations with
  float64 * The samples within a batch can currently influence each other a bit * We don’t perform
  the additional convergence confirmation * The success rate tracking changed a bit * Some other
  changes due to batching and merged loops
  
  *Parameters*
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) – Attack to use to
      find a starting points. Defaults to LinearSearchBlendedUniformNoiseAttack. Only used if
      starting_points is None.
    * **steps** (*int*) – Maximum number of steps to run. Might converge and stop before that.
    * **spherical_step** (*float*) – Initial step size for the orthogonal (spherical) step.
    * **source_step** (*float*) – Initial step size for the step towards the target.
    * **source_step_convergance** (*float*) – Sets the threshold of the stop criterion: if
      source_step becomes smaller than this value during the attack, the attack has converged and
      will stop.
    * **step_adaptation** (*float*) – Factor by which the step sizes are multiplied or divided.
    * **tensorboard** (*Union**[**typing_extensions.Literal**[**False**]**, **None**, **str**]*) –
      The log directory for TensorBoard summaries. If False, TensorBoard summaries will be disabled
      (default). If None, the logdir will be runs/CURRENT_DATETIME_HOSTNAME.
    * **update_stats_every_k** (*int*) –
  
  References
  
  *[12][129]*
    Wieland Brendel (*), Jonas Rauber (*), Matthias Bethge, “Decision-Based Adversarial Attacks:
    Reliable Attacks Against Black-Box Machine Learning Models”,
    [https://arxiv.org/abs/1712.04248][130]

* *class *foolbox.attacks.L0BrendelBethgeAttack(*init_attack=None*, *overshoot=1.1*, *steps=1000*,
*lr=0.001*, *lr_decay=0.5*, *lr_num_decay=20*, *momentum=0.8*, *tensorboard=False*,
*binary_search_steps=10*)[][131]*
  L0 variant of the Brendel & Bethge adversarial attack. [[#Bren19]_][132] This is a powerful
  gradient-based adversarial attack that follows the adversarial boundary (the boundary between the
  space of adversarial and non-adversarial images as defined by the adversarial criterion) to find
  the minimum distance to the clean image.
  
  This is the reference implementation of the Brendel & Bethge attack.
  
  References
  
  *13*
    Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan Ustyuzhaninov, Matthias Bethge,
    “Accurate, reliable and fast robustness evaluation”, 33rd Conference on Neural Information
    Processing Systems (2019) [https://arxiv.org/abs/1907.01003][133]
  
  *Parameters*
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) –
    * **overshoot** (*float*) –
    * **steps** (*int*) –
    * **lr** (*float*) –
    * **lr_decay** (*float*) –
    * **lr_num_decay** (*int*) –
    * **momentum** (*float*) –
    * **tensorboard** (*Union**[**typing_extensions.Literal**[**False**]**, **None**, **str**]*) –
    * **binary_search_steps** (*int*) –

* *class *foolbox.attacks.L1BrendelBethgeAttack(*init_attack=None*, *overshoot=1.1*, *steps=1000*,
*lr=0.001*, *lr_decay=0.5*, *lr_num_decay=20*, *momentum=0.8*, *tensorboard=False*,
*binary_search_steps=10*)[][134]*
  L1 variant of the Brendel & Bethge adversarial attack. [[#Bren19]_][135] This is a powerful
  gradient-based adversarial attack that follows the adversarial boundary (the boundary between the
  space of adversarial and non-adversarial images as defined by the adversarial criterion) to find
  the minimum distance to the clean image.
  
  This is the reference implementation of the Brendel & Bethge attack.
  
  References
  
  *14*
    Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan Ustyuzhaninov, Matthias Bethge,
    “Accurate, reliable and fast robustness evaluation”, 33rd Conference on Neural Information
    Processing Systems (2019) [https://arxiv.org/abs/1907.01003][136]
  
  *Parameters*
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) –
    * **overshoot** (*float*) –
    * **steps** (*int*) –
    * **lr** (*float*) –
    * **lr_decay** (*float*) –
    * **lr_num_decay** (*int*) –
    * **momentum** (*float*) –
    * **tensorboard** (*Union**[**typing_extensions.Literal**[**False**]**, **None**, **str**]*) –
    * **binary_search_steps** (*int*) –

* *class *foolbox.attacks.L2BrendelBethgeAttack(*init_attack=None*, *overshoot=1.1*, *steps=1000*,
*lr=0.001*, *lr_decay=0.5*, *lr_num_decay=20*, *momentum=0.8*, *tensorboard=False*,
*binary_search_steps=10*)[][137]*
  L2 variant of the Brendel & Bethge adversarial attack. [[#Bren19]_][138] This is a powerful
  gradient-based adversarial attack that follows the adversarial boundary (the boundary between the
  space of adversarial and non-adversarial images as defined by the adversarial criterion) to find
  the minimum distance to the clean image.
  
  This is the reference implementation of the Brendel & Bethge attack.
  
  References
  
  *15*
    Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan Ustyuzhaninov, Matthias Bethge,
    “Accurate, reliable and fast robustness evaluation”, 33rd Conference on Neural Information
    Processing Systems (2019) [https://arxiv.org/abs/1907.01003][139]
  
  *Parameters*
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) –
    * **overshoot** (*float*) –
    * **steps** (*int*) –
    * **lr** (*float*) –
    * **lr_decay** (*float*) –
    * **lr_num_decay** (*int*) –
    * **momentum** (*float*) –
    * **tensorboard** (*Union**[**typing_extensions.Literal**[**False**]**, **None**, **str**]*) –
    * **binary_search_steps** (*int*) –

* *class *foolbox.attacks.LinfinityBrendelBethgeAttack(*init_attack=None*, *overshoot=1.1*,
*steps=1000*, *lr=0.001*, *lr_decay=0.5*, *lr_num_decay=20*, *momentum=0.8*, *tensorboard=False*,
*binary_search_steps=10*)[][140]*
  L-infinity variant of the Brendel & Bethge adversarial attack. [[#Bren19]_][141] This is a
  powerful gradient-based adversarial attack that follows the adversarial boundary (the boundary
  between the space of adversarial and non-adversarial images as defined by the adversarial
  criterion) to find the minimum distance to the clean image.
  
  This is the reference implementation of the Brendel & Bethge attack.
  
  References
  
  *16*
    Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan Ustyuzhaninov, Matthias Bethge,
    “Accurate, reliable and fast robustness evaluation”, 33rd Conference on Neural Information
    Processing Systems (2019) [https://arxiv.org/abs/1907.01003][142]
  
  *Parameters*
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) –
    * **overshoot** (*float*) –
    * **steps** (*int*) –
    * **lr** (*float*) –
    * **lr_decay** (*float*) –
    * **lr_num_decay** (*int*) –
    * **momentum** (*float*) –
    * **tensorboard** (*Union**[**typing_extensions.Literal**[**False**]**, **None**, **str**]*) –
    * **binary_search_steps** (*int*) –

* *class *foolbox.attacks.L0FMNAttack(***, *steps=100*, *max_stepsize=1.0*, *min_stepsize=None*,
*gamma=0.05*, *init_attack=None*, *binary_search_steps=10*)[][143]*
  The L0 Fast Minimum Norm adversarial attack, in Lp norm. [17][144]
  
  *Parameters*
    * **steps** (*int*) – Number of iterations.
    * **max_stepsize** (*float*) – Initial stepsize for the gradient update.
    * **min_stepsize** (*Optional**[**float**]*) – Final stepsize for the gradient update. The
      stepsize will be reduced with a cosine annealing policy.
    * **gamma** (*float*) – Initial stepsize for the epsilon update. It will be updated with a
      cosine annealing reduction up to 0.001.
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) – Optional initial
      attack. If an initial attack is specified (or initial points are provided in the run), the
      attack will first try to search for the boundary between the initial point and the points in a
      class that satisfies the adversarial criterion.
    * **binary_search_steps** (*int*) – Number of steps to use for the search from the adversarial
      points. If no initial attack or adversarial starting point is provided, this parameter will be
      ignored.
  
  References
  
  *[17][145]*
    Maura Pintor, Fabio Roli, Wieland Brendel, Battista Biggio, “Fast Minimum-norm Adversarial
    Attacks through Adaptive Norm Constraints.” arXiv preprint arXiv:2102.12827 (2021).
    https://arxiv.org/abs/2102.12827

* *class *foolbox.attacks.L1FMNAttack(***, *steps=100*, *max_stepsize=1.0*, *min_stepsize=None*,
*gamma=0.05*, *init_attack=None*, *binary_search_steps=10*)[][146]*
  The L1 Fast Minimum Norm adversarial attack, in Lp norm. [18][147]
  
  *Parameters*
    * **steps** (*int*) – Number of iterations.
    * **max_stepsize** (*float*) – Initial stepsize for the gradient update.
    * **min_stepsize** (*Optional**[**float**]*) – Final stepsize for the gradient update. The
      stepsize will be reduced with a cosine annealing policy.
    * **gamma** (*float*) – Initial stepsize for the epsilon update. It will be updated with a
      cosine annealing reduction up to 0.001.
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) – Optional initial
      attack. If an initial attack is specified (or initial points are provided in the run), the
      attack will first try to search for the boundary between the initial point and the points in a
      class that satisfies the adversarial criterion.
    * **binary_search_steps** (*int*) – Number of steps to use for the search from the adversarial
      points. If no initial attack or adversarial starting point is provided, this parameter will be
      ignored.
  
  References
  
  *[18][148]*
    Maura Pintor, Fabio Roli, Wieland Brendel, Battista Biggio, “Fast Minimum-norm Adversarial
    Attacks through Adaptive Norm Constraints.” arXiv preprint arXiv:2102.12827 (2021).

* *class *foolbox.attacks.L2FMNAttack(***, *steps=100*, *max_stepsize=1.0*, *min_stepsize=None*,
*gamma=0.05*, *init_attack=None*, *binary_search_steps=10*)[][149]*
  The L2 Fast Minimum Norm adversarial attack, in Lp norm. [19][150]
  
  *Parameters*
    * **steps** (*int*) – Number of iterations.
    * **max_stepsize** (*float*) – Initial stepsize for the gradient update.
    * **min_stepsize** (*Optional**[**float**]*) – Final stepsize for the gradient update. The
      stepsize will be reduced with a cosine annealing policy.
    * **gamma** (*float*) – Initial stepsize for the epsilon update. It will be updated with a
      cosine annealing reduction up to 0.001.
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) – Optional initial
      attack. If an initial attack is specified (or initial points are provided in the run), the
      attack will first try to search for the boundary between the initial point and the points in a
      class that satisfies the adversarial criterion.
    * **binary_search_steps** (*int*) – Number of steps to use for the search from the adversarial
      points. If no initial attack or adversarial starting point is provided, this parameter will be
      ignored.
  
  References
  
  *[19][151]*
    Maura Pintor, Fabio Roli, Wieland Brendel, Battista Biggio, “Fast Minimum-norm Adversarial
    Attacks through Adaptive Norm Constraints.” arXiv preprint arXiv:2102.12827 (2021).
    https://arxiv.org/abs/2102.12827

* *class *foolbox.attacks.LInfFMNAttack(***, *steps=100*, *max_stepsize=1.0*, *min_stepsize=None*,
*gamma=0.05*, *init_attack=None*, *binary_search_steps=10*)[][152]*
  The L-infinity Fast Minimum Norm adversarial attack, in Lp norm. [20][153]
  
  *Parameters*
    * **steps** (*int*) – Number of iterations.
    * **max_stepsize** (*float*) – Initial stepsize for the gradient update.
    * **min_stepsize** (*Optional**[**float**]*) – Final stepsize for the gradient update. The
      stepsize will be reduced with a cosine annealing policy.
    * **gamma** (*float*) – Initial stepsize for the epsilon update. It will be updated with a
      cosine annealing reduction up to 0.001.
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) – Optional initial
      attack. If an initial attack is specified (or initial points are provided in the run), the
      attack will first try to search for the boundary between the initial point and the points in a
      class that satisfies the adversarial criterion.
    * **binary_search_steps** (*int*) – Number of steps to use for the search from the adversarial
      points. If no initial attack or adversarial starting point is provided, this parameter will be
      ignored.
  
  References
  
  *[20][154]*
    Maura Pintor, Fabio Roli, Wieland Brendel, Battista Biggio, “Fast Minimum-norm Adversarial
    Attacks through Adaptive Norm Constraints.” arXiv preprint arXiv:2102.12827 (2021).
    https://arxiv.org/abs/2102.12827

* *class *foolbox.attacks.PointwiseAttack(*init_attack=None*, *l2_binary_search=True*)[][155]*
  Starts with an adversarial and performs a binary search between the adversarial and the original
  for each dimension of the input individually. [21][156]
  
  References
  
  *[21][157]*
    Lukas Schott, Jonas Rauber, Matthias Bethge, Wieland Brendel, “Towards the first adversarially
    robust neural network model on MNIST”, [https://arxiv.org/abs/1805.09190][158]
  
  *Parameters*
    * **init_attack** (*Optional**[**foolbox.attacks.base.MinimizationAttack**]*) –
    * **l2_binary_search** (*bool*) –

* foolbox.attacks.FGM[][159]*
  alias of [`foolbox.attacks.fast_gradient_method.L2FastGradientAttack`][160]

* foolbox.attacks.FGSM[][161]*
  alias of [`foolbox.attacks.fast_gradient_method.LinfFastGradientAttack`][162]

* foolbox.attacks.L2PGD[][163]*
  alias of [`foolbox.attacks.projected_gradient_descent.L2ProjectedGradientDescentAttack`][164]

* foolbox.attacks.LinfPGD[][165]*
  alias of [`foolbox.attacks.projected_gradient_descent.LinfProjectedGradientDescentAttack`][166]

* foolbox.attacks.PGD[][167]*
  alias of [`foolbox.attacks.projected_gradient_descent.LinfProjectedGradientDescentAttack`][168]
[ Previous][169] [Next ][170]

© Copyright 2021, Jonas Rauber, Roland S. Zimmermann. Revision `1c55ee4d`.

Built with [Sphinx][171] using a [theme][172] provided by [Read the Docs][173].

[1]: https://github.com/bethgelab/foolbox/blob/1c55ee4d6847247eb50f34dd361ed5cd5b5a10bb/docs/modules
/attacks.rst
[2]: #module-foolbox.attacks
[3]: #module-foolbox.attacks
[4]: #foolbox.attacks.L2ContrastReductionAttack
[5]: #foolbox.attacks.VirtualAdversarialAttack
[6]: #foolbox.attacks.DDNAttack
[7]: #foolbox.attacks.L2ProjectedGradientDescentAttack
[8]: #foolbox.attacks.LinfProjectedGradientDescentAttack
[9]: #foolbox.attacks.L2BasicIterativeAttack
[10]: #foolbox.attacks.LinfBasicIterativeAttack
[11]: #foolbox.attacks.L2FastGradientAttack
[12]: #foolbox.attacks.LinfFastGradientAttack
[13]: #foolbox.attacks.L2AdditiveGaussianNoiseAttack
[14]: #foolbox.attacks.L2AdditiveUniformNoiseAttack
[15]: #foolbox.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack
[16]: #foolbox.attacks.L2ClippingAwareAdditiveUniformNoiseAttack
[17]: #foolbox.attacks.LinfAdditiveUniformNoiseAttack
[18]: #foolbox.attacks.L2RepeatedAdditiveGaussianNoiseAttack
[19]: #foolbox.attacks.L2RepeatedAdditiveUniformNoiseAttack
[20]: #foolbox.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack
[21]: #foolbox.attacks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack
[22]: #foolbox.attacks.LinfRepeatedAdditiveUniformNoiseAttack
[23]: #foolbox.attacks.InversionAttack
[24]: #foolbox.attacks.BinarySearchContrastReductionAttack
[25]: #foolbox.attacks.LinearSearchContrastReductionAttack
[26]: #foolbox.attacks.L2CarliniWagnerAttack
[27]: #foolbox.attacks.NewtonFoolAttack
[28]: #foolbox.attacks.EADAttack
[29]: #foolbox.attacks.GaussianBlurAttack
[30]: #foolbox.attacks.L2DeepFoolAttack
[31]: #foolbox.attacks.LinfDeepFoolAttack
[32]: #foolbox.attacks.SaltAndPepperNoiseAttack
[33]: #foolbox.attacks.LinearSearchBlendedUniformNoiseAttack
[34]: #foolbox.attacks.BinarizationRefinementAttack
[35]: #foolbox.attacks.DatasetAttack
[36]: #foolbox.attacks.BoundaryAttack
[37]: #foolbox.attacks.L0BrendelBethgeAttack
[38]: #foolbox.attacks.L1BrendelBethgeAttack
[39]: #foolbox.attacks.L2BrendelBethgeAttack
[40]: #foolbox.attacks.LinfinityBrendelBethgeAttack
[41]: #foolbox.attacks.L0FMNAttack
[42]: #foolbox.attacks.L1FMNAttack
[43]: #foolbox.attacks.L2FMNAttack
[44]: #foolbox.attacks.LInfFMNAttack
[45]: #foolbox.attacks.PointwiseAttack
[46]: #foolbox.attacks.FGM
[47]: #foolbox.attacks.L2FastGradientAttack
[48]: #foolbox.attacks.FGSM
[49]: #foolbox.attacks.LinfFastGradientAttack
[50]: #foolbox.attacks.L2PGD
[51]: #foolbox.attacks.L2ProjectedGradientDescentAttack
[52]: #foolbox.attacks.LinfPGD
[53]: #foolbox.attacks.LinfProjectedGradientDescentAttack
[54]: #foolbox.attacks.PGD
[55]: #foolbox.attacks.LinfProjectedGradientDescentAttack
[56]: #foolbox.attacks.L2ContrastReductionAttack
[57]: #foolbox.attacks.VirtualAdversarialAttack
[58]: #miy15
[59]: #miy15
[60]: #id1
[61]: #id2
[62]: https://arxiv.org/abs/1507.00677
[63]: #foolbox.attacks.DDNAttack
[64]: #rony18
[65]: #id3
[66]: https://arxiv.org/abs/1811.09600
[67]: #foolbox.attacks.L2ProjectedGradientDescentAttack
[68]: #foolbox.attacks.LinfProjectedGradientDescentAttack
[69]: #foolbox.attacks.L2BasicIterativeAttack
[70]: #foolbox.attacks.LinfBasicIterativeAttack
[71]: #foolbox.attacks.L2FastGradientAttack
[72]: #foolbox.attacks.LinfFastGradientAttack
[73]: #foolbox.attacks.L2AdditiveGaussianNoiseAttack
[74]: #foolbox.attacks.L2AdditiveUniformNoiseAttack
[75]: #foolbox.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack
[76]: #id30
[77]: https://arxiv.org/abs/2007.07677
[78]: #foolbox.attacks.L2ClippingAwareAdditiveUniformNoiseAttack
[79]: #id31
[80]: https://arxiv.org/abs/2007.07677
[81]: #foolbox.attacks.LinfAdditiveUniformNoiseAttack
[82]: #foolbox.attacks.L2RepeatedAdditiveGaussianNoiseAttack
[83]: #foolbox.attacks.L2RepeatedAdditiveUniformNoiseAttack
[84]: #foolbox.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack
[85]: #id32
[86]: https://arxiv.org/abs/2007.07677
[87]: #foolbox.attacks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack
[88]: #id33
[89]: https://arxiv.org/abs/2007.07677
[90]: #foolbox.attacks.LinfRepeatedAdditiveUniformNoiseAttack
[91]: #foolbox.attacks.InversionAttack
[92]: #hos16
[93]: #id11
[94]: https://arxiv.org/abs/1607.02533
[95]: distances.html#foolbox.distances.Distance
[96]: #foolbox.attacks.BinarySearchContrastReductionAttack
[97]: distances.html#foolbox.distances.Distance
[98]: #foolbox.attacks.LinearSearchContrastReductionAttack
[99]: distances.html#foolbox.distances.Distance
[100]: #foolbox.attacks.L2CarliniWagnerAttack
[101]: #carl16
[102]: #id12
[103]: https://arxiv.org/abs/1608.04644
[104]: #foolbox.attacks.NewtonFoolAttack
[105]: #jang17
[106]: #id13
[107]: https://dl.acm.org/citation.cfm?id=3134635
[108]: #foolbox.attacks.EADAttack
[109]: #chen18
[110]: #id14
[111]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16893
[112]: #foolbox.attacks.GaussianBlurAttack
[113]: distances.html#foolbox.distances.Distance
[114]: #foolbox.attacks.L2DeepFoolAttack
[115]: #moos15
[116]: #id15
[117]: https://arxiv.org/abs/1511.04599
[118]: #foolbox.attacks.LinfDeepFoolAttack
[119]: Seyed-MohsenMoosavi-Dezfooli,AlhusseinFawzi,PascalFrossard,"DeepFool:asimpleandaccuratemethod
tofooldeepneuralnetworks",https://arxiv.org/abs/1511.04599
[120]: #foolbox.attacks.SaltAndPepperNoiseAttack
[121]: #foolbox.attacks.LinearSearchBlendedUniformNoiseAttack
[122]: distances.html#foolbox.distances.Distance
[123]: #foolbox.attacks.BinarizationRefinementAttack
[124]: distances.html#foolbox.distances.Distance
[125]: #foolbox.attacks.DatasetAttack
[126]: distances.html#foolbox.distances.Distance
[127]: #foolbox.attacks.BoundaryAttack
[128]: #bren18
[129]: #id16
[130]: https://arxiv.org/abs/1712.04248
[131]: #foolbox.attacks.L0BrendelBethgeAttack
[132]: #id34
[133]: https://arxiv.org/abs/1907.01003
[134]: #foolbox.attacks.L1BrendelBethgeAttack
[135]: #id35
[136]: https://arxiv.org/abs/1907.01003
[137]: #foolbox.attacks.L2BrendelBethgeAttack
[138]: #id36
[139]: https://arxiv.org/abs/1907.01003
[140]: #foolbox.attacks.LinfinityBrendelBethgeAttack
[141]: #id37
[142]: https://arxiv.org/abs/1907.01003
[143]: #foolbox.attacks.L0FMNAttack
[144]: #pintor21l0
[145]: #id24
[146]: #foolbox.attacks.L1FMNAttack
[147]: #pintor21l1
[148]: #id25
[149]: #foolbox.attacks.L2FMNAttack
[150]: #pintor21l2
[151]: #id26
[152]: #foolbox.attacks.LInfFMNAttack
[153]: #pintor21linf
[154]: #id27
[155]: #foolbox.attacks.PointwiseAttack
[156]: #sch18
[157]: #id28
[158]: https://arxiv.org/abs/1805.09190
[159]: #foolbox.attacks.FGM
[160]: #foolbox.attacks.L2FastGradientAttack
[161]: #foolbox.attacks.FGSM
[162]: #foolbox.attacks.LinfFastGradientAttack
[163]: #foolbox.attacks.L2PGD
[164]: #foolbox.attacks.L2ProjectedGradientDescentAttack
[165]: #foolbox.attacks.LinfPGD
[166]: #foolbox.attacks.LinfProjectedGradientDescentAttack
[167]: #foolbox.attacks.PGD
[168]: #foolbox.attacks.LinfProjectedGradientDescentAttack
[169]: models.html
[170]: criteria.html
[171]: https://www.sphinx-doc.org/
[172]: https://github.com/readthedocs/sphinx_rtd_theme
[173]: https://readthedocs.org
