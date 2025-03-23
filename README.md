# learnSDE
Code and notes of learning SDE from scratch
## Theory
See [Theory](./paper/paper.pdf)

## NeuralSDE

1. On Neural Differential Equations [[paper]](https://arxiv.org/abs/2202.02435)
2. Scalable Gradients for Stochastic Differential Equations [[paper]](https://arxiv.org/abs/2001.01328) [[code]](https://github.com/google-research/torchsde)
3. Efficient and Accurate Gradients for Neural SDEs [[paper]](https://arxiv.org/abs/2105.13493) [[code]](https://github.com/patrick-kidger/torchcde) [[code]](https://github.com/google-research/torchsde)
4. Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit [[paper]](https://arxiv.org/abs/1905.09883)
5. Neural SDEs as Infinite-Dimensional GANs [[paper]](https://arxiv.org/abs/2102.03657) [[code]](https://github.com/google-research/torchsde)



## Model
|Model|Paper Link|Problem|Model|Loss Function|Details|Baseline|Dataset|
|---|---|---|---|---|---|---|---|
|**SPDE**|
|Physics-constrained deep learning for high-dimensional surrogate modeling and uncertainty quantification without labeled data|[[paper]](https://www.sciencedirect.com/science/article/pii/S0021999119303559?via%3Dihub)|
|Neural Stochastic Partial Differential Equations|[[paper]](https://arxiv.org/pdf/2110.10249v1.pdf)|学习SPDE解|FNO*|||DeepONet, FNO|随机 Ginzburg-Landau 方程, 随机 Korteweg–de Vries（KdV）方程, 随机 Navier-Stokes 方程|
|Simulator-free Solution of High-dimensional Stochastic Elliptic Partial Differential Equations using Deep Neural Networks| [[paper]](https://arxiv.org/pdf/1902.05200)|elliptic SPDE||||||
|Learning in Modal Space: Solving Time-Dependent Stochastic PDEs Using Physics-Informed Neural Networks|[[paper]](https://arxiv.org/pdf/1905.01205)|
|Deep Latent Regularity Network for Modeling Stochastic Partial Differential Equations|[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25938)
|Deep learning methods for stochastic Galerkin approximations of elliptic random PDEs.|[[paper]](https://arxiv.org/pdf/2409.08063)
|SOLVING STOCHASTIC PARTIAL DIFFERENTIAL EQUATIONS USING NEURAL NETWORKS IN THE WIENER CHAOS EXPANSION|[[paper]](https://arxiv.org/pdf/2411.03384)[[code]](https://github.com/psc25/ChaosSPDE)|
|Deep learning based numerical approximation algorithms for stochastic partial differential equations and high-dimensional nonlinear filtering problems|[[paper]](https://arxiv.org/abs/2012.01194)|
|A predictor-corrector deep learning algorithm for high dimensional stochastic partial differential equations|[[paper]](https://arxiv.org/abs/2208.09883)|
|Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations|[[paper]](https://arxiv.org/abs/1706.04702)|
|Neural variational Data Assimilation with Uncertainty Quantification using SPDE priors|[paper](https://arxiv.org/abs/2402.01855v3)|
|Neural SPDE solver for uncertainty quantification in high-dimensional space-time dynamics|[paper](https://arxiv.org/pdf/2311.01783)|


|Model|Paper Link|Problem|Model|Loss Function|Details|Baseline|Dataset|
|---|---|---|---|---|---|---|---|
|**SODE**|
|Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise|[[paper]](https://arxiv.org/abs/1906.02355)|改进Neural ODE, 增强噪声的对抗性||Path-wise Gradient Estimation|将 SDE 形式的噪声 引入 Neural ODE|Neural ODE, DropOut|CIFAR-10, STL-10, Tiny-ImageNet|
|Learning stochastic dynamics with statistics-informed neural network|[[paper]](https://www.sciencedirect.com/science/article/pii/S0021999122008828)|符合SDE的数据，做代理模型|LSTM|PDF+ACF|数据驱动，代理模型， 弱收敛非路径收敛|
|Implicit Stochastic Gradient Descent for Training Physics-informed Neural Networks|[[paper]](https://arxiv.org/pdf/2303.01767)|


## diffusion pde
GENERATIVE PDE CONTROL [[paper]](https://openreview.net/pdf?id=vaKnCahjdj)
