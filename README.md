# learnSDE
Code and notes of learning SDE from scratch
## Theory
See [Theory](./notebook/main.pdf)

## Stochastic Neural Network

- **On Neural Differential Equations**  
  [[paper]](https://arxiv.org/abs/2202.02435)
- **Scalable Gradients for Stochastic Differential Equations**  
  [[paper]](https://arxiv.org/abs/2001.01328) [[code]](https://github.com/google-research/torchsde)
- **Efficient and Accurate Gradients for Neural SDEs**  
  [[paper]](https://arxiv.org/abs/2105.13493) [[code]](https://github.com/patrick-kidger/torchcde) 
- **Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit**    
  [[paper]](https://arxiv.org/abs/1905.09883)
- **Neural SDEs as Infinite-Dimensional GANs**  
  [[paper]](https://arxiv.org/abs/2102.03657) [[code]](https://github.com/google-research/torchsde)


### SPDE

- **Physics-constrained deep learning for high-dimensional surrogate modeling and uncertainty quantification without labeled data**  
  [paper](https://www.sciencedirect.com/science/article/pii/S0021999119303559?via%3Dihub)

- **Neural Stochastic Partial Differential Equations**  
  [paper](https://arxiv.org/pdf/2110.10249v1.pdf)

- **Simulator-free Solution of High-dimensional Stochastic Elliptic Partial Differential Equations using Deep Neural Networks**  
  [paper](https://arxiv.org/pdf/1902.05200)

- **Learning in Modal Space: Solving Time-Dependent Stochastic PDEs Using Physics-Informed Neural Networks**  
  [paper](https://arxiv.org/pdf/1905.01205)

- **Deep Latent Regularity Network for Modeling Stochastic Partial Differential Equations**  
  [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25938)

- **Deep learning methods for stochastic Galerkin approximations of elliptic random PDEs**  
  [paper](https://arxiv.org/pdf/2409.08063)

- **Solving Stochastic Partial Differential Equations Using Neural Networks in the Wiener Chaos Expansion**  
  [paper](https://arxiv.org/pdf/2411.03384)  
  [code](https://github.com/psc25/ChaosSPDE)

- **Deep learning based numerical approximation algorithms for stochastic partial differential equations and high-dimensional nonlinear filtering problems**  
  [paper](https://arxiv.org/abs/2012.01194)

- **A predictor-corrector deep learning algorithm for high dimensional stochastic partial differential equations**  
  [paper](https://arxiv.org/abs/2208.09883)

- **Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations**  
  [paper](https://arxiv.org/abs/1706.04702)

- **Neural variational Data Assimilation with Uncertainty Quantification using SPDE priors**  
  [paper](https://arxiv.org/abs/2402.01855v3)

- **Neural SPDE solver for uncertainty quantification in high-dimensional space-time dynamics**  
  [paper](https://arxiv.org/pdf/2311.01783)

---

## SDE
- **Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise**  2019-2020CVPR
  [paper](https://arxiv.org/abs/192019-2020CVPR06.02355)

- **SDE-Net: Equipping Deep Neural Networks with Uncertainty Estimates.** ICML2020
  [[paper]](10.48550/arXiv.2008.10546)

- **Neural SDEs as Infinite-Dimensional GANs**  2021ICML
  [[paper]](https://arxiv.org/abs/2102.03657) [[code]](https://github.com/google-research/torchsde)

- **Efficient and Accurate Gradients for Neural SDEs**  NIPS2021
  [[paper]](https://arxiv.org/abs/2105.13493) [[code]](https://github.com/patrick-kidger/torchcde)

- **Learning stochastic dynamics with statistics-informed neural network**  JCP2023
  [paper](https://www.sciencedirect.com/science/article/pii/S0021999122008828)

- **Implicit Stochastic Gradient Descent for Training Physics-informed Neural Networks**  2023AAAI
  [paper](https://arxiv.org/pdf/2303.01767)

- **Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit**  
  [paper](https://arxiv.org/abs/1905.09883)


- **Theoretical guarantees for sampling and inference in generative models with latent diffusions**  

- **Neural Jump Stochastic Differential Equations**

- **Stochastic Normalizing Flows.**
- **Robust Pricing and Hedging via Neural SDEs.**
- **Scalable Gradients and Variational Inference for Stochastic Differential Equations.**




## Infinite Dimensional Diffusion
- **Stochastic Equations in Infinite Dimensions (Book)** 2014  
  [paper](https://www.cambridge.org/core/books/stochastic-equations-in-infinite-dimensions/6218FF6506BE364F82E3CF534FAC2FC5)

- **From Points to Functions: Infinite-Dimensional Representations in Diffusion Models** 2022  
  [paper](https://arxiv.org/pdf/2210.13774v1)
  [code](https://github.com/sarthmit/traj_drl)

- **Generative Models as Distributions of Functions** AISTATS2022    
  [paper](https://arxiv.org/pdf/2102.04776)
  [code](https://github.com/EmilienDupont/neural-function-distributions)


- **Generative Modelling with Inverse Heat Dissipation** 2023 ICLR2023  
  [paper](https://openreview.net/pdf?id=4PJUBT9f2Ol)
  [code](https://github.com/AaltoML/generative-inverse-heat-dissipation)
  
- **Infinite-Dimensional Diffusion Models** 2023 JMLR2024  
  [paper](https://www.jmlr.org/papers/volume25/23-1271/23-1271.pdf) 

- **Score-based Diffusion Models in Function Space** 2023-2025  
  [paper](https://arxiv.org/pdf/2302.07400)
  [code](https://github.com/lim0606/ddo)

- **Diffusion Generative Models in Infinite Dimensions**  2023 AISTATS2023  
  [paper](http://arxiv.org/abs/2212.00886)
  [code](https://github.com/GavinKerrigan/functional_diffusion)

- **Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation** 2023  
  [paper](http://arxiv.org/abs/2303.04772)
  [code](https://github.com/paullyonel/multileveldiff)



- **Continuous-Time Functional Diffusion Processes** NIPS2023  
  [paper](https://arxiv.org/pdf/2303.00800)

- **\infty Diff: Infinite resolution diffusion with subsampled mollified states** 2023 ICLR2024  
  [paper](https://arxiv.org/pdf/2303.18242)
  [code](https://github.com/samb-t/infty-diff)

- **Score-based Generative Modeling through Stochastic Evolution Equations in Hilbert Spaces** NIPS2023  
  [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/76c6f9f2475b275b92d03a83ea270af4-Paper-Conference.pdf)


## Infinite-dimensional Flow-based model

- **Functional Flow Matching** 2023  
  [paper](https://arxiv.org/pdf/2305.17209)
  [code](https://github.com/GavinKerrigan/functional_flow_matching)

- **Conditioning non-linear and infinite-dimensional diffusion processes**  NIPS2024  
  [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/14ad9256c430e6c8977e470d8e268320-Paper-Conference.pdf)

- **Stochastic Optimal Control for Diffusion Bridges in Function Spaces** NIPS2024  
  [paper](https://arxiv.org/pdf/2405.20630)
  [code](https://github.com/bw-park/DBFS)

- **Simulating Infinite-dimensional Nonlinear Diffusion Bridges** 2024  
  [paper](https://arxiv.org/pdf/2405.18353)
  [code](https://github.com/bookdiver/scoreoperator)

- **Probability-Flow ODE in Infinite-Dimensional Function Spaces** 2025  
  [paper](http://arxiv.org/abs/2503.10219)

## Generative Operator
- **Generative Adversarial Neural Operators** TMLR2022  
  [paper](https://arxiv.org/abs/2205.03017)
  [code](https://github.com/neuraloperator/GANO)

- **Variational Autoencoding Neural Operators** 2023 ICML
  [paper](https://proceedings.mlr.press/v202/seidman23a.html)

## Diffusion PDE

- **Generative PDE Control**   ICLR2024 workshop
  [paper](https://openreview.net/pdf?id=vaKnCahjdj)

- **DiffPhyCon: A Generative Approach to Control Complex Physical Systems** 2024 NIPS Oral
  [paper](http://arxiv.org/abs/2407.06494)

- **DiffusionPDE: Generative PDE-Solving Under Partial Observation** NIPS2024  
  [paper](https://arxiv.org/pdf/2406.17763)
  [code](https://github.com/jhhuangchloe/DiffusionPDE)

- **Diffusion-Based Inverse Solver on Function Spaces With Applications to PDEs** NIPS2024  workshop
  [paper](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_253.pdf)

- **Guided Diffusion Sampling on Function Spaces with Applications to PDEs** Maybe Underreview NIPS2025
  [paper](10.48550/arXiv.2505.17004)

- **FunDiff: Diffusion Models over Function Spaces for Physics-Informed Generative Modeling** 
  [paper](https://arxiv.org/pdf/2506.07902) 

- **A Denoising Diffusion Model for Fluid Field Prediction**
  [paper](http://arxiv.org/abs/2301.11661)


  
## MultiPhysics
- **M2PDE: Compositional Generative Multiphysics and Multi-component PDE Simulation** ICLR 2025  
  [paper](10.48550/arXiv.2412.04134)

##
- **Neural ODE**
- **Efficient and Accurate Gradients for Neural SDEs**