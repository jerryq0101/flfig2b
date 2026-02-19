from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

import flwr as fl
# Context is metadata to each client_fn
from flwr.common import Context, NDArrays, Scalar
# Conversion utility between flower and default python
from flwr.client import NumPyClient

from pyramidnet import PyramidNet
from dirichlet_split import dirichlet_label_split


# ----------------------------
# Model + Torch helpers
# ----------------------------
def get_model() -> nn.Module:
    return PyramidNet(dataset="cifar10", depth=110, alpha=84, num_classes=10, bottleneck=False)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weight getter for a particular model, converted into Flower format
def get_weights(model: nn.Module) -> NDArrays:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

# Uses a particular ND array from flower to populate model params
def set_weights(model: nn.Module, weights: NDArrays) -> None:
    keys = list(model.state_dict().keys())
    dev = next(model.parameters()).device
    state_dict = {}
    for k, w in zip(keys, weights):
        t = torch.from_numpy(w).to(dev)
        # Optional but recommended: match dtype of existing param/buffer
        t = t.to(dtype=model.state_dict()[k].dtype)
        state_dict[k] = t
    model.load_state_dict(state_dict, strict=True)


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

# Local training step in each client per round
def train_one_client(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> None:
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


# ----------------------------
# Flower client (fit only)
# ----------------------------
class FlowerClient(NumPyClient):
    def __init__(self, cid: int, train_loader: DataLoader, device: torch.device):
        # Each flower client has its data, device param, model copy
        self.cid = cid
        self.train_loader = train_loader
        self.device = device
        self.model = get_model().to(self.device)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_weights(self.model)

    # Client FL step on every communication round
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # Receive global model 
        set_weights(self.model, parameters)
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.05))
        # Local SGD step
        train_one_client(self.model, self.train_loader, self.device, epochs=local_epochs, lr=lr)
        # Return updated weights after train
        return get_weights(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Not used (we do server side evaluation)
        return 0.0, 0, {}


# ----------------------------
# Main
# ----------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = get_device()
    print("device =", device)
    if device.type == "cuda":
        print("gpu =", torch.cuda.get_device_name(0))

    # Normalization constants
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    print("A) building datasets...", flush=True)

    print("Downloading/loading CIFAR10...", flush=True)
    full_train_aug = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    full_train_eval = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_test)
    print("Loaded CIFAR10 train", flush=True)

    # deterministic 80/20 split on the *train* set
    n_total = len(full_train_aug)  # 50k
    n_test = int(0.2 * n_total)
    n_train = n_total - n_test

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_indices = np.array(perm[:n_train])
    test_indices = np.array(perm[n_train:])

    train_ds = Subset(full_train_aug, train_indices.tolist())
    test_ds = Subset(full_train_eval, test_indices.tolist())

    print("B) building loaders...", flush=True)

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    train_labels = np.array([full_train_aug.targets[i] for i in train_indices])

    # experimental parameters
    num_clients = 20
    clients_per_round = 20
    local_epochs = 1
    batch_size = 256
    rounds = 50
    alphas = [0.1, 1.0, 10.0, 100.0]

    # centralized baseline
    print("C) starting centralized baseline...", flush=True)

    base_model = get_model().to(device)
    train_loader_central = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    for _ in tqdm(range(rounds * local_epochs), desc="Centralized epochs"):
        train_one_client(base_model, train_loader_central, device, epochs=1, lr=0.05)
    central_acc = accuracy(base_model, test_loader, device)
    print("Centralized test acc:", central_acc)

    curves: Dict[float, List[float]] = {}

    # server-side evaluate_fn (checks accuracy of combined model every round)
    def make_evaluate_fn(test_loader: DataLoader, eval_device: torch.device, pbar: tqdm):
        def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
            # Get model
            model = get_model().to(eval_device)
            set_weights(model, parameters)
            # Evaluate accuracy of global model
            acc = accuracy(model, test_loader, eval_device)
            pbar.update(1)
            pbar.set_postfix(acc=f"{acc:.4f}")
            # print(f"[flower] round {server_round} acc={acc:.4f}", flush=True)
            return 0.0, {"acc": float(acc)}
        return evaluate_fn

    
    print("D) starting Flower simulation...", flush=True)

    for alpha in tqdm(alphas, desc="Alpha sessions"):
        
        # partition train_ds into client datasets (Dirichlet on labels)
        client_lists_local = dirichlet_label_split(train_labels, num_clients=num_clients, alpha=alpha, seed=0)

        client_subsets = []
        for k in range(num_clients):
            idx_positions = np.array(client_lists_local[k], dtype=int)
            subset_indices = train_indices[idx_positions]  # indices into full_train_aug
            client_subsets.append(Subset(full_train_aug, subset_indices.tolist()))

        client_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
            for ds in client_subsets
        ]

        # Use Single GPU, only spawn a single client on it at a time though.
        client_device = torch.device("cuda")

        def client_fn(context: Context):
            cid = int(context.node_config["partition-id"])
            return FlowerClient(cid, client_loaders[cid], client_device).to_client()

        def fit_config(server_round: int) -> Dict[str, Scalar]:
            return {"local_epochs": local_epochs, "lr": 0.05}

        pbar = tqdm(total=rounds, desc=f"Flower rounds (alpha={alpha})", leave=False)

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=clients_per_round / num_clients,
            fraction_evaluate=0.0,  # don't do client evaluation (div by 0 situation)
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            on_fit_config_fn=fit_config,
            evaluate_fn=make_evaluate_fn(test_loader, device, pbar),
        )

        print(f"\n=== alpha={alpha} starting Flower sim ===", flush=True)
        # Run simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 1.0},
            ray_init_args={"include_dashboard": False, "num_gpus": 1, "num_cpus": 1},
        )

        print(f"=== alpha={alpha} finished ===", flush=True)

        pbar.close()

        # Pull centralized acc history
        # history.metrics_centralized["acc"] is List[Tuple[round, acc]]
        accs = [v for _, v in history.metrics_centralized.get("acc", [])]
        curves[alpha] = accs

        print(f"alpha={alpha} final acc={curves[alpha][-1]:.4f}")


    # Plot
    xs = np.arange(0, rounds + 1)
    for alpha in alphas:
        plt.plot(xs, curves[alpha], label=f"alpha={alpha}")
    plt.axhline(central_acc, linestyle="--", label="Centralized")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Figure 2b reproduction (Flower)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2b_repro_flower.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    os.environ.setdefault("RAY_DISABLE_WINDOWS_JOB_OBJECTS_WARNING", "1")
    main()
