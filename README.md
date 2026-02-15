Simply open your conda environment with pytorch and `python fedavg_sim.py`

# Variable Notes on Paper reproduction
- Python 3.11.14
- PyTorch 2.6.0+cu124
- CUDA 12.4 (CUDA version of pytorch build)
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU (Driver: 576.02, CUDA version: 12.9)

`dirichlet_split.py` implements a Dirichlet label-skew partition of CIFAR-10 into client datasets.

`fedavg_sim.py` implements federated training with FedAvg, centralized version training, and generation of curves. PyramidNet-110 is used as the model architecture (ResNet-18 was used initially for debugging).

### FL simulation setup
No physical FL devices were available to me locally, so federated training was simulated using a sequential client loop.

I believe this has no impact on results as FedAvg depends only on local model weight updates, rather than real time parallelism.

### Federated learning configuration

These parameters were not specified in the paper for Figure 2b, and were chosen using the CIFAR10 baselines and held constant in the duration of the experiment.
- Number of clients: 20
- Clients per round: 20
- Local epochs per round: 1
- Batch size: 256

Client datasets were generated using Dirichlet label skew to match the paper's setting.

### Centralized Baseline
Since all clients participate each round and their datasets partition the full training set, one FL round processes approximately one epoch worth of data.

Therefore to make the centralized model result comparable, we train it for 50 epochs to match 50 FL rounds.

Note that the paper defines the centralized baseline as the converged centralized test accuracy. Our baseline is epoch matched rather than fully converged due to compute constraints.

### Evaluation protocol

The test dataset was fixed across all rounds so there are no random crops or flips, ensuring that ccuracy changes reflect model updates rather than randomness. 

### Model architecture
PyramidNet-110 (alpha = 84) was used following common CIFAR10 configurations from public implementation, as the paper did not specify PyramidNet exact parameters.

PyramidNet Source: https://github.com/dyhan0920/PyramidNet-PyTorch


### Optimizer 
SGD with momentum 0.9, weight decay 5e-4, learning rate 0.05.
(The optimizer and hyperparameters were not specified in the paper)

# Resulting Graph Notes

The qualitative trend from 2b is preserved: larger alpha yields higher accuracy, and accuracy improves over FL rounds. However, the absolute accuracies differ from the paper.

### Why FL in general is systematically lower relative to Figure 2b

Some training details affecting CIFAR-10 performance were not fully specified for this figure. For example, CIFAR baselines for PyramidNet usually use a learning rate decay schedule, while this reproduction used a constant learning rate of 0.05. This could cause FL to plateau earlier. Other choices like batch size, number of clients, and local epochs may also contribute.

### Why alpha=1.0 (or lower alpha in general) may be higher relatively
The paper’s results depend not only on the Dirichlet parameter alpha, but also on the number of clients K. 

In our experiment we used K = 20. If the paper used a larger amount of clients, each client would receive fewer samples. Under Dirichlet skew, smaller client datasets are more likely to miss some classes entirely, hurting FedAvg accuracy for smaller alpha. 

With fewer clients (=> larger data shards per client), each client is more likely to see examples from more classes, making the local updates less biased. This could have made alpha=1 in my graph perform better relative to the positioning of alpha=1 in the paper's graph.

