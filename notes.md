# Experimental Notes on Paper reproduction
- Python 3.11.14
- PyTorch 2.6.0+cu124
- CUDA 12.4 (CUDA version of pytorch build)
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU (Driver: 576.02, CUDA version: 12.9)

`dirichlet_split.py` implements a Dirichlet label-skew partition of CIFAR-10 into client datasets.

`fedavg_sim.py` implements federated training with FedAvg, centralized baseline training, and generation of Figure 2b-style accuracy curves. PyramidNet-110 is used as the model architecture (ResNet-18 was used initially for debugging).

### Simulation Setup
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


