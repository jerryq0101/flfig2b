# fedavg_sim.py
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
from pyramidnet import PyramidNet
import matplotlib.pyplot as plt

from dirichlet_split import dirichlet_label_split

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The Accuracy of the model
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

# Training a single client
# Client performs local SGD for E epochs.
def train_one_client(model, loader, device, epochs=1, lr=0.05, momentum=0.9, weight_decay=5e-4):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

# # ResNet 18 standing in for pyramid net (smaller)
# def get_model():
#     # ResNet18 baseline for CIFAR-10; adjust first conv for 32x32
#     model = models.resnet18(num_classes=10)
#     model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#     model.maxpool = nn.Identity()
#     return model

def get_model():
    return PyramidNet(dataset="cifar10", depth=110, alpha=84, num_classes=10, bottleneck=False)


# Fed average aggregation of client weights
def fedavg(state_dicts, weights):
    # weighted average of model parameters
    avg = {}
    total = float(sum(weights))
    for k in state_dicts[0].keys():
        avg[k] = sum(sd[k].float() * (w / total) for sd, w in zip(state_dicts, weights))
    return avg


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = get_device()

    print("device = ", device)
    if device.type == "cuda":
        print("gpu = ", torch.cuda.get_device_name(0))

    # --- Transforms ---
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),    #helps with training stablity
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # --- Two dataset objects, same underlying CIFAR-10 files ---
    full_train_aug = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    full_train_eval = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_test)

    # --- Deterministic 80/20 split by indices ---
    n_total = len(full_train_aug)  # 50000
    n_test = int(0.2 * n_total)
    n_train = n_total - n_test

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_indices = np.array(perm[:n_train])
    test_indices  = np.array(perm[n_train:])

    # --- Subsets: train uses augmented dataset, test uses eval dataset ---
    train_ds = Subset(full_train_aug, train_indices.tolist())
    test_ds  = Subset(full_train_eval, test_indices.tolist())

    # labels must come from the underlying dataset's targets using the train_indices
    train_labels = np.array([full_train_aug.targets[i] for i in train_indices])

    # DataLoader feeds the data to model in batches (256), faster than linear iteration
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # FL knobs (Non paper specified values, constant across rounds timeline)
    num_clients = 20    #20
    # num_clients = 5
    clients_per_round = 20  # just use all clients
    # clients_per_round = 5
    local_epochs = 1
    batch_size = 256
    # batch_size = 32

    # From Graph
    rounds = 50             # 50 total rounds (still need to record accuracy at before rounds)
    # rounds = 2
    # Alpha values in figure 2b
    alphas = [0.1, 1.0, 10.0, 100.0] # 0.1, 1, 10, 100
    # alphas = [1.0]
    curves = {}

    # Centralized baseline (dashed line)
    # TODO: Check if this thing needs to be plotted per different alpha value
    base_model = get_model().to(device)
    train_loader_central = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    for _ in tqdm(range(rounds * local_epochs), desc="Centralized epochs"):  # keep modest; adjust if you want closer to paper
        train_one_client(base_model, train_loader_central, device, epochs=1, lr=0.05)
    central_acc = evaluate(base_model, test_loader, device)
    print("Centralized test acc:", central_acc)

    for alpha in alphas:
        # Create client splits (Dirichlet label skew)
        # For 
        client_lists_local = dirichlet_label_split(train_labels, num_clients=num_clients, alpha=alpha, seed=0)

        # Convert label-indexed (0..len(train_ds)-1) indices into actual Subset indices
        # train_labels corresponds to train_ds.indices order, so client_lists_local references positions in train_ds
        client_subsets = []
        for k in range(num_clients):
            idx_positions = np.array(client_lists_local[k], dtype=int)
            subset_indices = train_indices[idx_positions]  # indices into full_train
            client_subsets.append(Subset(full_train_aug, subset_indices.tolist()))

        client_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
            for ds in client_subsets
        ]

        # Init global model
        global_model = get_model().to(device)
        # server model weights initial 
        global_state = deepcopy(global_model.state_dict())


        # Create per-client model objects ONCE (reuse each round)
        client_models = [get_model().to(device) for _ in range(num_clients)]


        # FL communication rounds loop

        # Accumulating accuracy for each round of a particular alpha
        accs = []
        acc0 = evaluate(global_model, test_loader, device)
        accs.append(acc0)

        for r in tqdm(range(1, rounds + 1), desc=f"FL alpha={alpha}"):
            # sample clients (here: all currently)
            selected = list(range(num_clients))[:clients_per_round]

            # Fake "parallelized" training for each client

            local_states = []
            local_weights = []
            for k in selected:
                # client_model = get_model().to(device)
                # # get the global state
                # client_model.load_state_dict(global_state)
                client_model = client_models[k]
                client_model.load_state_dict(global_state)

                loader = client_loaders[k]

                # Local training (for this client)
                train_one_client(client_model, loader, device, epochs=local_epochs, lr=0.05)

                # Add to local weight states
                # local_states.append(deepcopy(client_model.state_dict()))
                local_states.append({k: v.detach().clone() for k, v in client_model.state_dict().items()})

                # Also store this subset's size for the Fedavg
                local_weights.append(len(client_subsets[k]))

            # get the new fedavg weights
            global_state = fedavg(local_states, local_weights)
            # load in the new weights
            global_model.load_state_dict(global_state)

            # Evaluate accuracy for this round
            acc = evaluate(global_model, test_loader, device)
            accs.append(acc)

        curves[alpha] = accs
        print(f"alpha={alpha} final acc={accs[-1]:.4f}")

    # Plot like Fig 2b
    xs = np.arange(0, rounds + 1)
    for alpha in alphas:
        plt.plot(xs, curves[alpha], label=f"alpha={alpha}")
    plt.axhline(central_acc, linestyle="--", label="Centralized")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Figure 2b reproduction (trend)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2b_repro.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
