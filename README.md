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
|Learning stochastic dynamics with statistics-informed neural network|[[paper]](https://www.sciencedirect.com/science/article/pii/S0021999122008828)|符合SDE的数据，做代理模型|LSTM|PDF+ACF|数据驱动，代理模型， 弱收敛非路径收敛|
|Physics-constrained deep learning for high-dimensional surrogate modeling and uncertainty quantification without labeled data|[[paper]](https://www.sciencedirect.com/science/article/pii/S0021999119303559?via%3Dihub)|
|Neural Stochastic Partial Differential Equations|[[paper]](https://arxiv.org/pdf/2110.10249v1.pdf)|学习SPDE解|FNO*|||DeepONet, FNO|随机 Ginzburg-Landau 方程, 随机 Korteweg–de Vries（KdV）方程, 随机 Navier-Stokes 方程|
|Simulator-free Solution of High-dimensional Stochastic Elliptic Partial Differential Equations using Deep Neural Networks| [[paper]](https://arxiv.org/pdf/1902.05200)|elliptic SPDE||||||
|Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise|[[paper]](https://arxiv.org/abs/1906.02355)|改进Neural ODE, 增强噪声的对抗性||Path-wise Gradient Estimation|将 SDE 形式的噪声 引入 Neural ODE|Neural ODE, DropOut|CIFAR-10, STL-10, Tiny-ImageNet|

